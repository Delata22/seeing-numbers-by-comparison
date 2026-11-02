"""Trains the encoder + the two heads, then computes all evaluations.

Usage (from the repository root): uv run python run.py --seed 0
Writes results/seed<k>/metrics.json and results/seed<k>/z.npz (for the PCA).
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAIRES_RETENUES = {(3, 6), (6, 3), (5, 8), (8, 5), (2, 7), (7, 2)}
DIM_Z = 64


class Encodeur(nn.Module):
    """Shared CNN: 64x64 image -> vector z (64 numbers)."""
    def __init__(self):
        super().__init__()
        def bloc(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.conv = nn.Sequential(
            bloc(1, 32, 1), bloc(32, 32, 2),    # 32x32
            bloc(32, 64, 1), bloc(64, 64, 2),   # 16x16
            bloc(64, 128, 2),                   # 8x8
            bloc(128, 128, 2),                  # 4x4
        )
        self.proj = nn.Linear(128 * 16, DIM_Z)

    def forward(self, x):
        return self.proj(self.conv(x).flatten(1))


class Modele(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encodeur()
        # "how many?" head ("combien"): z -> 9 scores (classes 1..9)
        self.combien = nn.Sequential(nn.Linear(DIM_Z, 128), nn.ReLU(), nn.Linear(128, 9))
        # "which one is larger?" head: (z1, z2) -> 2 scores (0 = left, 1 = right)
        self.compare = nn.Sequential(nn.Linear(2 * DIM_Z, 128), nn.ReLU(), nn.Linear(128, 2))

    def comparer(self, z1, z2):
        return self.compare(torch.cat([z1, z2], 1))


def charger(nom, dev):
    d = np.load(f"data/{nom}.npz")
    X = torch.tensor(d["X"], dtype=torch.float32, device=dev).unsqueeze(1) / 255.0
    return X, torch.tensor(d["y"], device=dev), d["mode"]


def augmenter(x):
    """Random 90-degree rotations and flips: they do not change the quantity."""
    k = np.random.randint(4)
    x = torch.rot90(x, k, (2, 3))
    return x.flip(3) if np.random.rand() < 0.5 else x


@torch.no_grad()
def encoder(m, X, bs=2000):
    return torch.cat([m.enc(X[i:i + bs]) for i in range(0, len(X), bs)])


@torch.no_grad()
def evaluer(m, X, y, mode, rng, n_paires=1000, n_triplets=20000):
    m.eval()
    z = encoder(m, X)
    pred = m.combien(z).argmax(1) + 1
    yc = y.cpu().numpy()
    pc = pred.cpu().numpy()
    res = {}

    # 1. "How many?" accuracy per quantity (and per mode: fixed area / fixed size)
    res["combien"] = {str(n): float((pc[yc == n] == n).mean()) for n in range(1, 10)}
    res["combien_par_mode"] = {
        nom: {str(n): float((pc[(yc == n) & (mode == k)] == n).mean()) for n in range(1, 10)}
        for k, nom in [(0, "aire"), (1, "taille")]}
    conf = np.zeros((9, 9), int)
    np.add.at(conf, (yc - 1, pc - 1), 1)
    res["confusion"] = conf.tolist()

    # 2-3. Comparison on all ordered pairs (a, b), a != b, unseen images
    idx = {n: np.where(yc == n)[0] for n in range(1, 10)}
    paires = {}
    for a in range(1, 10):
        for b in range(1, 10):
            if a == b:
                continue
            i = torch.tensor(rng.choice(idx[a], n_paires), device=z.device)
            j = torch.tensor(rng.choice(idx[b], n_paires), device=z.device)
            p = m.comparer(z[i], z[j]).argmax(1).cpu().numpy()       # 1 = right is larger
            verite = int(b > a)
            # control: compare the two answers of the "how many?" head
            via_combien = (pc[j.cpu().numpy()] > pc[i.cpu().numpy()]).astype(int)
            # symmetry: swapping left and right must flip the answer
            p_inv = m.comparer(z[j], z[i]).argmax(1).cpu().numpy()
            paires[f"{a}-{b}"] = {
                "acc": float((p == verite).mean()),
                "acc_via_combien": float((via_combien == verite).mean()),
                "symetrie": float((p != p_inv).mean()),
                "ratio": min(a, b) / max(a, b),
                "retenue": (a, b) in PAIRES_RETENUES,
            }
    res["paires"] = paires

    # 4. Transitivity: 3 images with distinct quantities, 3 comparisons.
    # Inconsistent = cycle (A>B, B>C and yet C>A).
    ns = np.array([rng.choice(np.arange(1, 10), 3, replace=False) for _ in range(n_triplets)])
    ii = [torch.tensor([rng.choice(idx[n]) for n in ns[:, k]], device=z.device) for k in range(3)]

    def plus_grand(u, v):  # True if image u is judged larger than v
        return m.comparer(z[u], z[v]).argmax(1).cpu().numpy() == 0
    ab, bc, ca = plus_grand(ii[0], ii[1]), plus_grand(ii[1], ii[2]), plus_grand(ii[2], ii[0])
    cycle = (ab == bc) & (bc == ca)
    touche_retenue = np.array([any((ns[t, k], ns[t, (k + 1) % 3]) in PAIRES_RETENUES
                                   for k in range(3)) for t in range(n_triplets)])
    res["transitivite"] = {
        "coherence": float(1 - cycle.mean()),
        "coherence_avec_paire_retenue": float(1 - cycle[touche_retenue].mean()),
        "coherence_sans_paire_retenue": float(1 - cycle[~touche_retenue].mean()),
        "tout_juste": float(((ab == (ns[:, 0] > ns[:, 1])) & (bc == (ns[:, 1] > ns[:, 2]))
                             & (ca == (ns[:, 2] > ns[:, 0]))).mean()),
    }
    return res, z.cpu().numpy(), pc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(1000 + args.seed)
    dev = "cuda"
    out = f"results/seed{args.seed}"
    os.makedirs(out, exist_ok=True)

    X, y, _ = charger("train_disque", dev)
    par_classe = len(X) // 9   # images are sorted by quantity: 1, 1, ..., 2, 2, ...
    assert (y.view(9, par_classe) == torch.arange(1, 10, device=dev)[:, None]).all()
    paires_ok = [(a, b) for a in range(1, 10) for b in range(1, 10)
                 if a != b and (a, b) not in PAIRES_RETENUES]
    print(f"{len(paires_ok)} paires ordonnées à l'entraînement, {len(PAIRES_RETENUES)} retenues")

    m = Modele().to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.steps)
    t0 = time.time()
    for step in range(args.steps):
        m.train()
        # "how many?" task
        i = torch.randint(len(X), (args.bs,), device=dev)
        # "comparison" task: an allowed pair, then one image of each quantity
        choix = np.random.randint(len(paires_ok), size=args.bs)
        a = torch.tensor([paires_ok[c][0] for c in choix], device=dev)
        b = torch.tensor([paires_ok[c][1] for c in choix], device=dev)
        ia = (a - 1) * par_classe + torch.randint(par_classe, (args.bs,), device=dev)
        ib = (b - 1) * par_classe + torch.randint(par_classe, (args.bs,), device=dev)
        x = augmenter(torch.cat([X[i], X[ia], X[ib]]))
        z = m.enc(x)
        zc, za, zb = z[:args.bs], z[args.bs:2 * args.bs], z[2 * args.bs:]
        l_combien = F.cross_entropy(m.combien(zc), y[i] - 1)
        cible = (b > a).long()
        l_compare = F.cross_entropy(m.comparer(za, zb), cible)
        loss = l_combien + l_compare
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"pas {step:5d}  perte combien {l_combien.item():.3f}  "
                  f"perte comparaison {l_compare.item():.3f}  ({time.time() - t0:.0f} s)")
    torch.save(m.state_dict(), f"{out}/modele.pt")

    metrics = {"duree_s": time.time() - t0}
    zs = {}
    for forme in ["disque", "carre", "triangle"]:
        Xt, yt, mt = charger(f"test_{forme}", dev)
        r, z, pc = evaluer(m, Xt, yt, mt, rng)
        metrics[forme] = r
        zs[f"z_{forme}"] = z; zs[f"y_{forme}"] = yt.cpu().numpy(); zs[f"mode_{forme}"] = mt
        zs[f"pred_{forme}"] = pc
        acc = np.mean(list(r["combien"].values()))
        ret = np.mean([v["acc"] for v in r["paires"].values() if v["retenue"]])
        print(f"[{forme}] combien {acc:.3f} | paires retenues {ret:.3f} | "
              f"transitivité {r['transitivite']['coherence']:.3f}")
    np.savez_compressed(f"{out}/z.npz", **zs)
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
