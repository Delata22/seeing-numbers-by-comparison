"""Numbers above 20 known only through sums (evaluation only, no training).

Models: results/expressions/D{3,9}_{general,succ}_seed{0,1,2} (series 1, "+", totals up to 60 seen only as sums,
no image with more than 20 objects). Plus, for A3, the starting comparison models (range 0-3 and 0-9, seed 2).
Image line: mean of z per n over the test images 0..20 (both modes), fitted by z ~ k n + b, then extrapolated.

A1. For T = 0..100: 200 random decompositions into terms in 0..20 (2-3 terms up to 20, 3 terms up to 60,
    5 terms beyond), test images drawn at random; s_hat = value computed by the brick.
    Offset (in steps) between the mean of s_hat and the extrapolated line; spread of s_hat (in steps).
A2. Sum vs sum, 4 or 5 terms per side, totals 61..100 (never seen): exact "=", exact +/-1, and side.
A3. Fixed-area images with 21 to 30 objects (never seen): (i) reading on the 0..20 line; (ii) image of n vs a
    sum with the same total (2 terms): exact "="; vs a sum with total n +/- 1: side. Also on the starting models
    (reading, and n vs n+1 between images). Caveat: at 30 objects with fixed area, the shapes are tiny.

Usage (from the repository root): uv run python au_dela_20.py -> results/au_dela_20.json and logs/au_dela_20.txt
"""
import glob, json, os
import numpy as np
import torch

from compare import GAUCHE, EGAL, DROITE, cible
from reduction import Comparateur, encoder
from addition import charger_pool
from expressions import Brique, decomposer

dev = "cuda"
torch.manual_seed(0)
Xt, yt, _, _, _ = charger_pool(["test_plage15", "test_plage16a20"], dev)
it = {n: np.where(yt == n)[0] for n in range(21)}
g = np.load("data/grand_mix.npz")
Xg = torch.tensor(g["X"], dtype=torch.uint8, device=dev).unsqueeze(1)
ig = {n: np.where((g["y"] == n) & (g["mode"] == 0))[0] for n in range(21, 31)}


@torch.no_grad()
def ligne(m):
    z = encoder(m, Xt)[:, 0]
    moy = np.array([z[torch.tensor(it[n], device=dev)].mean().item() for n in range(21)])
    k, b = np.polyfit(np.arange(21), moy, 1)
    return z, k, b


@torch.no_grad()
def valeur(h, z, listes, rng):
    L = max(len(x) for x in listes)
    T = np.zeros((len(listes), L), int); Ln = np.array([len(x) for x in listes])
    for i, x in enumerate(listes):
        T[i, :len(x)] = x
    idx = np.vectorize(lambda n: rng.choice(it[n]))(T)
    Z = z[torch.tensor(idx, device=dev)]
    return h.valeur(Z, torch.zeros_like(Z, dtype=torch.long), torch.tensor(Ln, device=dev))


def decomposer_sur(T, L, rng, tmax=20):
    """L terms in 0..tmax summing to T: balanced start, then random transfers (works even close to L * tmax)."""
    t = np.full(L, T // L); t[: T % L] += 1
    for _ in range(30):
        i, j = rng.choice(L, 2, replace=False)
        q = int(rng.integers(0, min(t[i], tmax - t[j]) + 1))
        t[i] -= q; t[j] += q
    return [int(x) for x in rng.permutation(t)]


def decomps(T, nb, rng):
    L = (2 if T <= 10 else 3) if T <= 20 else (3 if T <= 60 else 5)
    return [decomposer_sur(T, L, rng) for _ in range(nb)]


@torch.no_grad()
def a3_lecture(m, z_tr, k, b, rng):
    zg = encoder(m, Xg)[:, 0]
    lect = {n: float((zg[torch.tensor(ig[n], device=dev)].mean().item() - b) / k) for n in range(21, 31)}
    cote = {}
    for n in range(21, 30):
        i = torch.tensor(rng.choice(ig[n], 300), device=dev); j = torch.tensor(rng.choice(ig[n + 1], 300), device=dev)
        l1 = m.comparer(zg[i][:, None], zg[j][:, None]).float(); l2 = m.comparer(zg[j][:, None], zg[i][:, None]).float()
        cote[n] = float(((l1[:, DROITE] > l1[:, GAUCHE]).float().mean() + (l2[:, GAUCHE] > l2[:, DROITE]).float().mean()).item() / 2)
    return zg, lect, cote


res = {}
L_txt = []
# starting models (A3 only)
for N in [3, 9]:
    rng = np.random.default_rng(N)
    m = Comparateur(8, 1, 3, 128).to(dev); m.load_state_dict(torch.load(f"modeles/plage_N0-{N}_x1-8_z1_tete128_egal/seed2.pt")); m.eval()
    z, k, b = ligne(m)
    _, lect, cote = a3_lecture(m, z, k, b, rng)
    res[f"depart_0-{N}"] = {"A3": {"lecture": lect, "n_n1_cote_images": cote}}
    L_txt.append(f"départ 0-{N} | A3 lecture 21..30 : " + " ".join(f"{lect[n]:.1f}" for n in range(21, 31))
                 + " | n/n+1 côté (images) : " + " ".join(f"{cote[n]:.2f}" for n in range(21, 30)))

for d in sorted(glob.glob("results/expressions/D*_seed*")):
    nom = os.path.basename(d); rng = np.random.default_rng(1)
    m = Comparateur(8, 1, 3, 128).to(dev); m.load_state_dict(torch.load(f"{d}/modele.pt")); m.eval()
    h = Brique(n_ops=1).to(dev); h.load_state_dict(torch.load(f"{d}/h.pt")); h.eval()
    z, k, b = ligne(m)
    r = {"k": float(k), "b": float(b)}
    # A1
    a1 = {}
    with torch.no_grad():
        for T in list(range(0, 101, 5)) + [21, 25, 30, 40, 50, 60, 61, 70, 80, 90, 100]:
            s = valeur(h, z, decomps(T, 200, rng), rng)[:, 0].cpu().numpy()
            a1[T] = {"ecart_pas": float((s.mean() - (k * T + b)) / k), "lu": float((s.mean() - b) / k),
                     "dispersion_pas": float(s.std() / abs(k))}
    r["A1"] = {str(T): v for T, v in sorted(a1.items())}
    # A2
    with torch.no_grad():
        p_all, c_all, tg_all, td_all = [], [], [], []
        while len(tg_all) < 3000:
            Tg = int(rng.integers(61, 96)); dlt = int(rng.choice([0, 0, -1, 1, -2, 2, -5, 5])); Td = Tg + dlt
            Lg = max(int(rng.integers(4, 6)), -(-Tg // 20)); Ld = max(int(rng.integers(4, 6)), -(-Td // 20))
            gd = decomposer_sur(Tg, Lg, rng); dd = decomposer_sur(Td, Ld, rng)
            if gd is None or dd is None:
                continue
            tg_all.append((gd, dd))
        G = [x[0] for x in tg_all]; D = [x[1] for x in tg_all]
        l = m.comparer(valeur(h, z, G, rng), valeur(h, z, D, rng)).float()
        tg = np.array([sum(x) for x in G]); td = np.array([sum(x) for x in D])
        p = l.argmax(1).cpu().numpy(); cote = ((l[:, GAUCHE] > l[:, DROITE]).cpu().numpy() == (tg > td))
        dd_ = np.abs(tg - td)
        r["A2"] = {"egal": float((p[dd_ == 0] == EGAL).mean()),
                   "ecart1_exact": float((p[dd_ == 1] == cible(tg[dd_ == 1], td[dd_ == 1])).mean()),
                   "ecart1_cote": float(cote[dd_ == 1].mean()),
                   "ecart2_cote": float(cote[dd_ == 2].mean()), "ecart5_cote": float(cote[dd_ == 5].mean())}
    # A3
    with torch.no_grad():
        zg, lect, cote_img = a3_lecture(m, z, k, b, rng)
        eg, cote_s = {}, {}
        for n in range(21, 31):
            sm = [decomposer_sur(n, 2, rng) for _ in range(300)]
            s = valeur(h, z, sm, rng)
            i = torch.tensor(rng.choice(ig[n], 300), device=dev)
            eg[n] = float((m.comparer(zg[i][:, None], s).argmax(1) == EGAL).float().mean())
            c_ = []
            for dlt in [-1, 1]:
                sm2 = [decomposer_sur(n + dlt, 2, rng) for _ in range(300)]
                s2 = valeur(h, z, sm2, rng)
                l = m.comparer(zg[i][:, None], s2).float()
                c_.append(float(((l[:, GAUCHE] > l[:, DROITE]) == (dlt < 0)).float().mean()))
            cote_s[n] = float(np.mean(c_))
        r["A3"] = {"lecture": lect, "n_n1_cote_images": cote_img, "image_vs_somme_egal": eg, "image_vs_somme_pm1_cote": cote_s}
    res[nom] = r
    L_txt.append(f"{nom} | A1 lu (écart, dispersion) : " + " ".join(f"{T}→{a1[T]['lu']:.1f}({a1[T]['ecart_pas']:+.1f},{a1[T]['dispersion_pas']:.1f})" for T in [10, 20, 30, 40, 60, 80, 100])
                 + f" | A2 61-100 : = {r['A2']['egal']:.2f}, ±1 exact {r['A2']['ecart1_exact']:.2f} côté {r['A2']['ecart1_cote']:.2f}, ±5 côté {r['A2']['ecart5_cote']:.2f}"
                 + " | A3 lecture : " + " ".join(f"{lect[n]:.1f}" for n in (21, 25, 30))
                 + " ; image = somme : " + " ".join(f"{eg[n]:.2f}" for n in (21, 25, 30))
                 + " ; ±1 côté : " + " ".join(f"{cote_s[n]:.2f}" for n in (21, 25, 30)))
    print(L_txt[-1], flush=True)
json.dump(res, open("results/au_dela_20.json", "w"), indent=1)
open("logs/au_dela_20.txt", "w").write("\n".join(L_txt) + "\n")
print("\n".join(L_txt[:2]))
