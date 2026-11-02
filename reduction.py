"""Progressive network shrinking: how far can we compress while keeping 100 % on 1-9?

Same setup as compare.py: 64x64, mix version, condition A (3-6, 5-8, 2-7 never seen, all
equalities seen), with or without the "equal" class. What changes:
  - adjustable width: channels 32-32-64-64-128-128 divided by --largeur (width: 1, 2, 4, 8, 16...);
  - adjustable size of z (--dim_z); comparison head with --tete hidden units (128 by default);
  - channels: max(1, channel // width) -> x1/32 = 1-1-2-2-4-4, x1/64 = 1-1-1-1-2-2;
  - learning rate: linear warm-up over 200 steps, then constant (instead of OneCycle);
  - early stopping: validation loss (test images, allowed pairs) measured every
    100 steps. Stop after 5 measurements without improvement, cap of 4000 steps, and keep the best
    checkpoint. "Improvement" = a drop of more than 1 %; once the loss is below 0.001, a drop of more
    than 0.0005 (otherwise the cross-entropy loss keeps decreasing in relative terms forever);
  - generalization to 10-20 (fixed size) and 10-30 (fixed area) measured at 200, 500, 1000 steps and at the stop.

Usage (from the repository root): uv run python reduction.py --largeur 4 --dim_z 16 --seed 0 [--sans_egal]
Writes results/reduction/L<largeur>_z<dim_z>_<eg|se>_seed<s>/{metrics.json, modele.pt}
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from run import augmenter
from compare import (PAIRES_RETENUES, GAUCHE, EGAL, DROITE, cible, charger, flottant, analyser)

PATIENCE, EVAL_TOUS, PLAFOND, MONTEE = 5, 100, 4000, 200
PAS_GENERALISATION = {200, 500, 1000}


class Encodeur(nn.Module):
    def __init__(self, largeur=1, dim_z=16):
        super().__init__()
        c = [max(1, x // largeur) for x in (32, 32, 64, 64, 128, 128)]
        s = [1, 2, 1, 2, 2, 2]
        couches, cin = [], 1
        for co, st in zip(c, s):
            couches += [nn.Conv2d(cin, co, 3, st, 1), nn.BatchNorm2d(co), nn.ReLU()]
            cin = co
        self.conv = nn.Sequential(*couches)
        self.proj = nn.Linear(c[-1] * 16, dim_z)

    def forward(self, x):
        return self.proj(self.conv(x).flatten(1))


class Comparateur(nn.Module):
    def __init__(self, largeur=1, dim_z=16, n_sorties=3, tete=128):
        super().__init__()
        self.n_sorties = n_sorties
        self.enc = Encodeur(largeur, dim_z)
        self.tete = nn.Sequential(nn.Linear(2 * dim_z, tete), nn.ReLU(), nn.Linear(tete, n_sorties))

    def comparer(self, z1, z2):
        l = self.tete(torch.cat([z1, z2], 1))
        if self.n_sorties == 2:
            l = torch.cat([l[:, :1], torch.full_like(l[:, :1], -1e4), l[:, 1:]], 1)
        return l


@torch.no_grad()
def encoder(m, X, bs=1000):
    return torch.cat([m.enc(flottant(X[i:i + bs])).float() for i in range(0, len(X), bs)])


@torch.no_grad()
def perte_validation(m, z, i, j, t):
    l = m.comparer(z[i], z[j]).float()
    return F.cross_entropy(l, t).item(), (l.argmax(1) == t).float().mean().item()


@torch.no_grad()
def generalisation(m, Xg, yg, modeg, rng):
    """Readout of the z -> n line (fitted on 1-9) and comparisons beyond 9."""
    m.eval()
    z = encoder(m, Xg)
    zn = z.cpu().numpy()
    petit = yg <= 9
    sc = StandardScaler().fit(zn[petit])
    pr = LinearRegression().fit(sc.transform(zn[petit]), yg[petit]).predict(sc.transform(zn))
    out = {}
    for k, nom, hi in [(1, "taille", 20), (0, "aire", 30)]:
        idx = {n: np.where((yg == n) & (modeg == k))[0] for n in range(1, hi + 1)}
        lect = {n: float(pr[idx[n]].mean()) for n in range(1, hi + 1)}
        # arbitrary pairs a != b in 10..hi
        a = rng.integers(10, hi + 1, 6000); b = rng.integers(10, hi + 1, 6000)
        garde = a != b; a, b = a[garde], b[garde]
        i = np.array([rng.choice(idx[x]) for x in a]); j = np.array([rng.choice(idx[x]) for x in b])
        l = m.comparer(z[i], z[j]).float()
        cote = ((l[:, GAUCHE] > l[:, DROITE]).cpu().numpy() == (a > b)).mean()
        exact = (l.argmax(1).cpu().numpy() == cible(a, b)).mean()
        # n versus n+1
        eg, co = [], []
        for n in range(10, hi):
            i = rng.choice(idx[n], 200); j = rng.choice(idx[n + 1], 200)
            l1, l2 = m.comparer(z[i], z[j]).float(), m.comparer(z[j], z[i]).float()
            eg.append(((l1.argmax(1) == EGAL).float().mean() + (l2.argmax(1) == EGAL).float().mean()).item() / 2)
            co.append(((l1[:, DROITE] > l1[:, GAUCHE]).float().mean()
                       + (l2[:, GAUCHE] > l2[:, DROITE]).float().mean()).item() / 2)
        out[nom] = {"lecture": lect, "bon_cote": float(cote), "exact": float(exact),
                    "egal_n_n1": float(np.mean(eg)), "bon_cote_n_n1": float(np.mean(co))}
    m.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--largeur", type=int, default=1)
    ap.add_argument("--dim_z", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sans_egal", action="store_true")
    ap.add_argument("--tete", type=int, default=128, help="hidden units of the comparison head")
    ap.add_argument("--bs", type=int, default=384)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    nom = f"L{args.largeur}_z{args.dim_z}_{'se' if args.sans_egal else 'eg'}_seed{args.seed}"
    if args.tete != 128:
        nom = nom.replace("_seed", f"_t{args.tete}_seed")
    out = f"results/reduction/{nom}"
    os.makedirs(out, exist_ok=True)

    X, y, _ = charger("train_mix", dev)
    par_classe = len(X) // 9
    inegales = np.array([(a, b) for a in range(1, 10) for b in range(1, 10)
                         if a != b and (a, b) not in PAIRES_RETENUES])
    egales = np.array([(n, n) for n in range(1, 10)]) if not args.sans_egal else None
    n_eg = 0 if args.sans_egal else args.bs // 3

    # Validation: test images, allowed pairs, fixed draw
    Xv, yv, _ = charger("test_mix", dev)
    yvn = yv.cpu().numpy()
    rv = np.random.default_rng(500 + args.seed)
    nv = 6000
    pv = [inegales[rv.integers(len(inegales), size=nv - (0 if args.sans_egal else nv // 3))]]
    if not args.sans_egal:
        pv.insert(0, egales[rv.integers(9, size=nv // 3)])
    pv = np.concatenate(pv)
    idxv = {n: np.where(yvn == n)[0] for n in range(1, 10)}
    iv = torch.tensor([rv.choice(idxv[a]) for a in pv[:, 0]], device=dev)
    jv = torch.tensor([rv.choice(idxv[b]) for b in pv[:, 1]], device=dev)
    tv = torch.tensor(cible(pv[:, 0], pv[:, 1]), device=dev)

    g = np.load("data/grand_mix.npz")
    Xg = torch.tensor(g["X"], dtype=torch.uint8, device=dev).unsqueeze(1)
    yg, modeg = g["y"], g["mode"]
    rg = np.random.default_rng(700 + args.seed)

    m = Comparateur(args.largeur, args.dim_z, 2 if args.sans_egal else 3, args.tete).to(dev)
    n_enc = sum(p.numel() for p in m.enc.parameters()); n_tete = sum(p.numel() for p in m.tete.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / MONTEE))
    print(f"{nom} : encodeur {n_enc} paramètres, tête {n_tete}", flush=True)

    suivi, gen = [], {}
    meilleur, meilleur_pas, etat, sans_progres, pas_100 = float("inf"), 0, None, 0, None
    t0 = time.time()
    for step in range(1, PLAFOND + 1):
        m.train()
        tirage = [inegales[np.random.randint(len(inegales), size=args.bs - n_eg)]]
        if n_eg:
            tirage.insert(0, egales[np.random.randint(9, size=n_eg)])
        ab = np.concatenate(tirage)
        a = torch.tensor(ab[:, 0], device=dev); b = torch.tensor(ab[:, 1], device=dev)
        ia = (a - 1) * par_classe + torch.randint(par_classe, (args.bs,), device=dev)
        ib = (b - 1) * par_classe + torch.randint(par_classe, (args.bs,), device=dev)
        t = torch.tensor(cible(ab[:, 0], ab[:, 1]), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(augmenter(flottant(torch.cat([X[ia], X[ib]]))))
            logits = m.comparer(z[:args.bs], z[args.bs:])
        loss = F.cross_entropy(logits.float(), t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if step in PAS_GENERALISATION:
            gen[step] = generalisation(m, Xg, yg, modeg, rg)
        if step % EVAL_TOUS == 0:
            m.eval()
            zv = encoder(m, Xv)
            pv_loss, pv_acc = perte_validation(m, zv, iv, jv, tv)
            suivi.append({"pas": step, "perte_val": pv_loss, "acc_val": pv_acc, "s": time.time() - t0})
            if pv_acc == 1.0 and pas_100 is None:
                pas_100 = step
            seuil = 0.01 * meilleur if meilleur >= 1e-3 else 5e-4
            if meilleur - pv_loss > seuil or meilleur == float("inf"):
                meilleur, meilleur_pas, sans_progres = pv_loss, step, 0
                etat = copy.deepcopy(m.state_dict())
            else:
                sans_progres += 1
            if step % 500 == 0:
                print(f"pas {step:5d}  perte val {pv_loss:.4f}  exactitude val {pv_acc:.4f}  "
                      f"({time.time() - t0:.0f} s)", flush=True)
            if sans_progres >= PATIENCE:
                break
    pas_arret = step
    duree = time.time() - t0
    m.load_state_dict(etat)
    torch.save(m.state_dict(), f"{out}/modele.pt")
    print(f"arrêt au pas {pas_arret} (meilleur point : pas {meilleur_pas}, perte val {meilleur:.5f}), "
          f"100 % en validation au pas {pas_100}, {duree:.0f} s", flush=True)

    gen["arret"] = generalisation(m, Xg, yg, modeg, rg)
    res, zs = analyser(m, dev, set(), np.random.default_rng(1000 + args.seed), ("mix",))
    zt, yt = zs["z_mix"], zs["y_mix"]
    acp = PCA(1).fit(zt).explained_variance_ratio_[0] if zt.shape[1] > 1 else 1.0
    p = res["mix"]["paires"]
    resume = {
        "comparaison": float(np.mean([v["acc"] for v in p.values()])),
        "inegales": float(np.mean([v["acc"] for k, v in p.items() if k.split("-")[0] != k.split("-")[1]])),
        "retenues": float(np.mean([v["acc"] for v in p.values() if v["retenue"]])),
        "egales": float(np.mean([v["acc"] for k, v in p.items() if k.split("-")[0] == k.split("-")[1]])),
        "transitivite": res["mix"]["transitivite"]["coherence"],
        "sonde_classif": res["sondes"]["mix"]["classif"],
        "sonde_r2": res["sondes"]["mix"]["regression_r2"],
        "acp_axe1": float(acp),
    }
    metrics = {"nom": nom, "largeur": args.largeur, "dim_z": args.dim_z, "sans_egal": args.sans_egal,
               "seed": args.seed, "tete": args.tete, "params_encodeur": n_enc, "params_tete": n_tete,
               "pas_arret": pas_arret, "meilleur_pas": meilleur_pas, "pas_100_val": pas_100,
               "duree_s": duree, "s_par_pas": duree / pas_arret, "suivi": suivi,
               "resume": resume, "generalisation": {str(k): v for k, v in gen.items()}, "complet": res}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)
    ga, gt = gen["arret"]["aire"], gen["arret"]["taille"]
    print(f"RÉSULTAT {nom} | comparaison {resume['comparaison']:.3f} inégales {resume['inegales']:.3f} "
          f"retenues {resume['retenues']:.3f} transitivité {resume['transitivite']:.3f} sonde R² {resume['sonde_r2']:.3f} | "
          f"taille 10-20 bon côté {gt['bon_cote']:.3f} « égal » n/n+1 {gt['egal_n_n1']:.2f} lit 20 → {gt['lecture'][20]:.1f} | "
          f"aire 10-30 bon côté {ga['bon_cote']:.3f} lit 30 → {ga['lecture'][30]:.1f} | arrêt {pas_arret}", flush=True)


if __name__ == "__main__":
    main()
