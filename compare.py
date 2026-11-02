"""Comparison only: two images -> left larger / equal / right larger.

The network is never given a number label. After training, linear probes
on z (frozen encoder) measure whether the number can be read out of z.

Usage (from the repository root): uv run python compare.py --data mix --seed 0
        (condition B by default: 3-6, 5-8, 2-7 and 6 = 6 never seen; condition A,
        where all equalities are seen, is no longer used)
Images: mixed shapes (disk, square, triangle drawn at random for each object),
for both training and test.
Writes results/<data>_<condition>_seed<k>/{metrics.json, z.npz, modele.pt}.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler

from run import Encodeur, DIM_Z, augmenter


def charger(nom, dev):
    """Images kept on the GPU as 0-255 integers (uint8): 4 times less memory than floats.
    They are converted to float (0-1) only when fed to the network."""
    d = np.load(f"data/{nom}.npz")
    X = torch.tensor(d["X"], dtype=torch.uint8, device=dev).unsqueeze(1)
    return X, torch.tensor(d["y"], device=dev), d["mode"]


def flottant(x):
    return x.float() / 255.0


def formes_images(nom, n):
    """Shape of each image (0, 1, 2 in the uni version; -1 if mixed or if the information is missing)."""
    d = np.load(f"data/{nom}.npz")
    return d["forme"] if "forme" in d.files else -np.ones(n, dtype=int)


@torch.no_grad()
def encoder(m, X, bs=500):
    return torch.cat([m.enc(flottant(X[i:i + bs])) for i in range(0, len(X), bs)])

PAIRES_RETENUES = {(3, 6), (6, 3), (5, 8), (8, 5), (2, 7), (7, 2)}
EGAL_RETENU = {"A": set(), "B": {6}}
GAUCHE, EGAL, DROITE = 0, 1, 2


class Comparateur(nn.Module):
    """n_sorties = 3 (left / equal / right) or 2 (left / right, --sans_egal option).
    With 2 outputs, a very low "equal" score is inserted and never chosen: this is exactly a
    2-class classification, and all the evaluation code (which expects 3 scores) works unchanged."""
    def __init__(self, n_sorties=3):
        super().__init__()
        self.n_sorties = n_sorties
        self.enc = Encodeur()
        self.tete = nn.Sequential(nn.Linear(2 * DIM_Z, 128), nn.ReLU(), nn.Linear(128, n_sorties))

    def comparer(self, z1, z2):
        l = self.tete(torch.cat([z1, z2], 1))
        if self.n_sorties == 2:
            bas = torch.full_like(l[:, :1], -1e4)
            l = torch.cat([l[:, :1], bas, l[:, 1:]], 1)
        return l


def charger_modele(chemin, dev):
    sd = torch.load(chemin, map_location=dev)
    m = Comparateur(sd["tete.2.weight"].shape[0]).to(dev)
    m.load_state_dict(sd)
    return m


def cible(a, b):
    return np.where(a > b, GAUCHE, np.where(a == b, EGAL, DROITE))


# ---------------------------------------------------------------- evaluations
@torch.no_grad()
def eval_comparaison(m, z, y, egal_retenu, rng, fimg=None, n_paires=1000, n_triplets=20000):
    """fimg: shape of each image (0, 1, 2) for the "uni" version, None otherwise."""
    idx = {n: np.where(y == n)[0] for n in range(1, 10)}
    ok_meme, ok_inter = [], []
    res = {}
    paires = {}
    for a in range(1, 10):
        for b in range(1, 10):
            i = rng.choice(idx[a], n_paires)
            j = rng.choice(idx[b], n_paires)
            if a == b:  # two different images with the same quantity
                j = np.where(i == j, rng.choice(idx[b], n_paires), j)
            logits = m.comparer(z[i], z[j])
            p = logits.argmax(1).cpu().numpy()
            p_inv = m.comparer(z[j], z[i]).argmax(1).cpu().numpy()
            if fimg is not None:
                meme = fimg[i] == fimg[j]
                ok_meme.append((p == cible(a, b))[meme]); ok_inter.append((p == cible(a, b))[~meme])
            miroir = {GAUCHE: DROITE, EGAL: EGAL, DROITE: GAUCHE}
            paires[f"{a}-{b}"] = {
                "acc": float((p == cible(a, b)).mean()),
                "repart": [float((p == k).mean()) for k in range(3)],  # left / equal / right
                "symetrie": float(np.mean([miroir[u] == v for u, v in zip(p, p_inv)])),
                "ratio": min(a, b) / max(a, b),
                "retenue": (a, b) in PAIRES_RETENUES or (a == b and a in egal_retenu),
            }
    res["paires"] = paires
    if fimg is not None:  # same-shape pairs vs different-shape pairs
        res["acc_meme_forme"] = float(np.concatenate(ok_meme).mean())
        res["acc_interforme"] = float(np.concatenate(ok_inter).mean())

    # Transitivity on 3 images with distinct quantities.
    # "u > v" if the left score exceeds the right score. Inconsistent = cycle.
    ns = np.array([rng.choice(np.arange(1, 10), 3, replace=False) for _ in range(n_triplets)])
    ii = [np.array([rng.choice(idx[n]) for n in ns[:, k]]) for k in range(3)]

    def plus_grand(u, v):
        l = m.comparer(z[u], z[v])
        return (l[:, GAUCHE] > l[:, DROITE]).cpu().numpy()
    ab, bc, ca = plus_grand(ii[0], ii[1]), plus_grand(ii[1], ii[2]), plus_grand(ii[2], ii[0])
    cycle = (ab == bc) & (bc == ca)
    touche = np.array([any((ns[t, k], ns[t, (k + 1) % 3]) in PAIRES_RETENUES for k in range(3))
                       for t in range(n_triplets)])
    res["transitivite"] = {
        "coherence": float(1 - cycle.mean()),
        "coherence_avec_paire_retenue": float(1 - cycle[touche].mean()),
    }
    return res


def sondes(z_tr, y_tr, cibles_test):
    """Linear probes fitted on z_tr, tested on each dataset in cibles_test.

    - 9-class linear classification (logistic regression);
    - linear regression z -> n, then rounding (is the number a line in z?).
    """
    sc = StandardScaler().fit(z_tr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(sc.transform(z_tr), y_tr)
    reg = LinearRegression().fit(sc.transform(z_tr), y_tr)
    out = {}
    for nom, (z, y) in cibles_test.items():
        zs = sc.transform(z)
        pc = clf.predict(zs)
        pr = reg.predict(zs)
        prr = np.clip(np.rint(pr), 1, 9)
        out[nom] = {
            "classif_par_n": {str(n): float((pc[y == n] == n).mean()) for n in range(1, 10)},
            "classif": float((pc == y).mean()),
            "regression_arrondie": float((prr == y).mean()),
            "regression_r2": float(1 - ((pr - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "regression_moyenne_par_n": {str(n): float(pr[y == n].mean()) for n in range(1, 10)},
        }
    return out


def sonde_blanc(z_tr, b_tr, z_te, b_te):
    """Control: can the amount of white (pixels) be read linearly from z?"""
    sc = StandardScaler().fit(z_tr)
    reg = LinearRegression().fit(sc.transform(z_tr), b_tr)
    pr = reg.predict(sc.transform(z_te))
    return float(1 - ((pr - b_te) ** 2).sum() / ((b_te - b_te.mean()) ** 2).sum())


def analyser(m, dev, egal_retenu, rng, formes=("mix",), comparaisons=True):
    """All measurements for a model (trained or not). The 1st dataset is used to fit the probes.

    Comparisons: the only train/test difference is the PAIRS. The main result
    (key = version name) is therefore measured on the training images: each image has already been
    seen, but never in a held-out pair. Secondary result on unseen images (key *_images_neuves).
    """
    m.eval()
    jeux = {}
    for forme in formes:
        X, y, mode = charger(f"test_{forme}", dev)
        z = encoder(m, X)
        blanc = X.sum((1, 2, 3)).float().cpu().numpy() / 255.0
        fimg = formes_images(f"test_{forme}", len(y))
        jeux[forme] = (z, y.cpu().numpy(), mode, blanc, fimg)
    # Probes: half of the 1st test set to fit, the other half (+ other datasets) to test
    z_d, y_d, _, b_d, f_d = jeux[formes[0]]
    zn = z_d.cpu().numpy()
    perm = rng.permutation(len(y_d))
    tr, te = perm[: len(perm) // 2], perm[len(perm) // 2:]
    cibles = {formes[0]: (zn[te], y_d[te])}
    for k, nom in enumerate(["disque", "carre", "triangle"]):  # "uni" version: one probe per shape
        if (f_d[te] == k).any():
            cibles[f"{formes[0]}_{nom}"] = (zn[te][f_d[te] == k], y_d[te][f_d[te] == k])
    for f in formes[1:]:
        cibles[f] = (jeux[f][0].cpu().numpy(), jeux[f][1])
    res = {"sondes": sondes(zn[tr], y_d[tr], cibles),
           "sonde_blanc_r2": sonde_blanc(zn[tr], b_d[tr], zn[te], b_d[te])}
    for f, (z, y, _, _, fimg) in jeux.items():
        if not comparaisons:
            break
        res[f + "_images_neuves"] = eval_comparaison(m, z, y, egal_retenu, rng,
                                                     fimg if (fimg >= 0).all() else None)
        Xtr, ytr, _ = charger(f"train_{f}", dev)
        ftr = formes_images(f"train_{f}", len(ytr))
        res[f] = eval_comparaison(m, encoder(m, Xtr), ytr.cpu().numpy(), egal_retenu, rng,
                                  ftr if (ftr >= 0).all() else None)
        del Xtr
    zs = {}
    for f, (z, y, mode, blanc, _) in jeux.items():
        zs[f"z_{f}"] = z.cpu().numpy(); zs[f"y_{f}"] = y; zs[f"mode_{f}"] = mode; zs[f"blanc_{f}"] = blanc
    return res, zs


@torch.no_grad()
def point_trajectoire(m, Xd, yd, rng):
    """Quick measurement during training: comparison and linear probe on the test set."""
    m.eval()
    z = encoder(m, Xd)
    zn, y = z.cpu().numpy(), yd.cpu().numpy()
    i, j = rng.integers(len(y), size=4000), rng.integers(len(y), size=4000)
    p = m.comparer(z[i], z[j]).argmax(1).cpu().numpy()
    perm = rng.permutation(len(y)); tr, te = perm[: len(y) // 2], perm[len(y) // 2:]
    s = sondes(zn[tr], y[tr], {"t": (zn[te], y[te])})["t"]
    m.train()
    return {"comparaison": float((p == cible(y[i], y[j])).mean()),
            "sonde": s["classif"], "sonde_par_n": s["classif_par_n"],
            "regression_r2": s["regression_r2"]}


# ---------------------------------------------------------------- training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", choices=["A", "B"], default="B")
    ap.add_argument("--data", choices=["mix", "uni"], default="mix")
    ap.add_argument("--eval_seulement", action="store_true",
                    help="no training: reload modele.pt and redo all measurements")
    ap.add_argument("--sans_egal", action="store_true",
                    help="2 classes (left / right), no equal pairs during training")
    ap.add_argument("--deux_sens", action="store_true",
                    help="each sampled pair is also shown in reverse order (6<7 and 7>6)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=384)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(1000 + args.seed)
    dev = "cuda"
    etiquette = args.condition + ("2s" if args.deux_sens else "") + ("se" if args.sans_egal else "")
    out = f"results/{args.data}_{etiquette}_seed{args.seed}"
    os.makedirs(out, exist_ok=True)
    egal_retenu = EGAL_RETENU[args.condition]

    X, y, _ = charger(f"train_{args.data}", dev)
    par_classe = len(X) // 9
    assert (y.view(9, par_classe) == torch.arange(1, 10, device=dev)[:, None]).all()
    inegales = np.array([(a, b) for a in range(1, 10) for b in range(1, 10)
                         if a != b and (a, b) not in PAIRES_RETENUES])
    egales = np.array([(n, n) for n in range(1, 10) if n not in egal_retenu and not args.sans_egal])
    print(f"condition {args.condition} : {len(inegales)} paires inégales, "
          f"{len(egales)} égalités à l'entraînement")

    m = Comparateur(2 if args.sans_egal else 3).to(dev)
    # Control: the same network, untrained (same measurements)
    m0 = Comparateur(2 if args.sans_egal else 3).to(dev)
    m0.load_state_dict(m.state_dict())

    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=args.steps)
    n_eg = 0 if args.sans_egal else args.bs // 3
    Xd, yd, _ = charger(f"test_{args.data}", dev)
    pas_mesure = {0, 5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, args.steps - 1}
    trajectoire = {}
    t0 = time.time()
    if args.eval_seulement:
        m.load_state_dict(torch.load(f"{out}/modele.pt"))
        print("évaluation seule : modèle rechargé (pas de trajectoire d'entraînement)")
    for step in range(0 if args.eval_seulement else args.steps):
        if step in pas_mesure:
            trajectoire[step] = point_trajectoire(m, Xd, yd, np.random.default_rng(step))
        m.train()
        tirage = [inegales[np.random.randint(len(inegales), size=args.bs - n_eg)]]
        if n_eg:
            tirage.insert(0, egales[np.random.randint(len(egales), size=n_eg)])
        ab = np.concatenate(tirage)
        a = torch.tensor(ab[:, 0], device=dev); b = torch.tensor(ab[:, 1], device=dev)
        ia = (a - 1) * par_classe + torch.randint(par_classe, (args.bs,), device=dev)
        ib = (b - 1) * par_classe + torch.randint(par_classe, (args.bs,), device=dev)
        t = torch.tensor(cible(ab[:, 0], ab[:, 1]), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(augmenter(flottant(torch.cat([X[ia], X[ib]]))))
            za, zb = z[:args.bs], z[args.bs:]
            logits = m.comparer(za, zb)
            if args.deux_sens:  # same image pair in the other order: left <-> right (0 <-> 2)
                logits = torch.cat([logits, m.comparer(zb, za)])
                t = torch.cat([t, 2 - t])
        loss = F.cross_entropy(logits.float(), t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0 or step == args.steps - 1:
            acc = (logits.argmax(1) == t).float().mean().item()
            print(f"pas {step:5d}  perte {loss.item():.3f}  exactitude du lot {acc:.3f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
    if not args.eval_seulement:
        torch.save(m.state_dict(), f"{out}/modele.pt")

    res, zs = analyser(m, dev, egal_retenu, rng, (args.data,))
    res0, zs0 = analyser(m0, dev, egal_retenu, np.random.default_rng(2000 + args.seed), (args.data,),
                         comparaisons=False)
    metrics = {"data": args.data, "condition": etiquette, "seed": args.seed, "duree_s": time.time() - t0,
               "entraine": res, "trajectoire": trajectoire, "non_entraine": {"sondes": res0["sondes"],
                                                 "sonde_blanc_r2": res0["sonde_blanc_r2"]}}
    np.savez_compressed(f"{out}/z.npz", **zs, **{f"init_{k}": v for k, v in zs0.items() if k.startswith("z_")})
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)

    s = res["sondes"]; s0 = res0["sondes"]
    for f in [args.data]:
        p = res[f]["paires"]
        ret = np.mean([v["acc"] for k, v in p.items() if v["retenue"] and k.split("-")[0] != k.split("-")[1]])
        print(f"[{f}] comparaison {np.mean([v['acc'] for v in p.values()]):.3f} | "
              f"paires inégales retenues {ret:.3f} | 6=6 {p['6-6']['acc']:.3f} | "
              f"transitivité {res[f]['transitivite']['coherence']:.3f} | "
              f"sonde linéaire {s[f]['classif']:.3f} (non entraîné {s0[f]['classif']:.3f})")
        if "acc_interforme" in res[f]:
            print(f"   même forme {res[f]['acc_meme_forme']:.3f} | formes différentes {res[f]['acc_interforme']:.3f} | "
                  + " | ".join(f"sonde {k.split('_')[1]} {v['classif']:.3f}" for k, v in s.items() if "_" in k))


if __name__ == "__main__":
    main()
