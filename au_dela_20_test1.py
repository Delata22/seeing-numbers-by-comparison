"""Test 1 of the "above 20" series: after fine-tuning on images 0..30 (part B), is an image of n objects
(21..30) judged equal to a sum [a] + [b] with the same total?

Models: results/au_dela_20_B/D<N>_<type>_seed<s>.pt (24 models). For "addition" and "succ", the brick h is the one
of the starting expression model (unchanged by fine-tuning); the expression model BEFORE fine-tuning is also
measured, for comparison. "base" and "controle" have no brick: image vs image only.

For each n = 21..30 (grand_mix test images, fixed area):
- image vs image: reading on the 0..20 line (extrapolated), exact "=" between two different images of n,
  correct side for n vs n+1;
- image vs sum (two terms in 0..20, test images 0..20): exact "=" if a + b = n; correct side and exact
  answer if a + b = n +/- 1. Both orders (image on the left, then on the right), averaged;
- offset (in steps) between the image line and the sum line: (mean of s_hat - mean of image z) / k.

Usage (from the repository root): uv run python au_dela_20_test1.py -> results/au_dela_20_test1.json and logs/au_dela_20_test1.txt
"""
import glob, json, os, sys
import numpy as np
import torch

from compare import GAUCHE, EGAL, DROITE, cible
from reduction import Comparateur, encoder
from expressions import Brique
from addition import charger_pool

dev = "cuda"
# same data and functions as au_dela_20.py (copied: that file runs its evaluation on import)
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
    t = np.full(L, T // L); t[: T % L] += 1
    for _ in range(30):
        i, j = rng.choice(L, 2, replace=False)
        q = int(rng.integers(0, min(t[i], tmax - t[j]) + 1))
        t[i] -= q; t[j] += q
    return [int(x) for x in rng.permutation(t)]
NS = range(21, 31)
NB = 300


@torch.no_grad()
def comparer_deux_ordres(m, za, zb, ta, tb):
    """Compare za (total ta) with zb (total tb) in both orders: exact answer and correct side."""
    ex, co = [], []
    for g, d, tg, td in [(za, zb, ta, tb), (zb, za, tb, ta)]:
        l = m.comparer(g, d).float()
        p = l.argmax(1).cpu().numpy()
        ex.append((p == cible(tg, td)).mean())
        if (tg != td).any():
            co.append((((l[:, GAUCHE] > l[:, DROITE]).cpu().numpy()) == (tg > td)).mean())
    return float(np.mean(ex)), (float(np.mean(co)) if co else None)


@torch.no_grad()
def evaluer(m, h, rng, succ=False, oubli=False):
    m.eval()
    z, k, b = ligne(m)
    zg = encoder(m, Xg)[:, 0]
    r = {"k": float(k), "b": float(b), "par_n": {}}
    for n in NS:
        i = torch.tensor(rng.choice(ig[n], NB), device=dev); i2 = torch.tensor(rng.choice(ig[n], NB), device=dev)
        zi = zg[i][:, None]
        d = {"lecture": float((zg[torch.tensor(ig[n], device=dev)].mean().item() - b) / k)}
        tn = np.full(NB, n)
        d["img_img_egal"], _ = comparer_deux_ordres(m, zi, zg[i2][:, None], tn, tn)
        if n < 30:
            j = torch.tensor(rng.choice(ig[n + 1], NB), device=dev)
            d["img_img_n_n1_exact"], d["img_img_n_n1_cote"] = comparer_deux_ordres(m, zi, zg[j][:, None], tn, tn + 1)
        if h is not None:
            s = valeur(h, z, [decomposer_sur(n, 2, rng) for _ in range(NB)], rng)
            d["img_somme_egal"], _ = comparer_deux_ordres(m, zi, s, tn, tn)
            d["ecart_lignes_pas"] = float((s.mean().item() - zg[torch.tensor(ig[n], device=dev)].mean().item()) / k)
            d["lecture_somme"] = float((s.mean().item() - b) / k)
            ex, co = [], []
            for dlt in [-1, 1]:
                s2 = valeur(h, z, [decomposer_sur(n + dlt, 2, rng) for _ in range(NB)], rng)
                e_, c_ = comparer_deux_ordres(m, zi, s2, tn, tn + dlt)
                ex.append(e_); co.append(c_)
            d["img_somme_pm1_exact"], d["img_somme_pm1_cote"] = float(np.mean(ex)), float(np.mean(co))
        r["par_n"][str(n)] = d
    if h is not None:   # check within the already known range: image of n (test 0..20) vs sum of n, n = 10, 15, 20
        r["dans_plage"] = {}
        for n in (10, 15, 20):
            zi = z[torch.tensor(rng.choice(it[n], NB), device=dev)][:, None]
            s = valeur(h, z, [decomposer_sur(n, 2, rng) for _ in range(NB)], rng)
            tn = np.full(NB, n)
            e_, _ = comparer_deux_ordres(m, zi, s, tn, tn)
            r["dans_plage"][str(n)] = {"img_somme_egal": e_, "ecart_lignes_pas": float((s.mean().item() - zi.mean().item()) / k)}
    if h is not None and succ:   # chained successors: image of n vs [20] + [1] * (n - 20); within the range, [10] + [1] * (n - 10)
        r["succ_chaine"] = {}
        for n in list(range(21, 31)) + [15, 20]:
            base_ = 20 if n > 20 else 10
            zi = (zg[torch.tensor(rng.choice(ig[n], NB), device=dev)] if n > 20 else z[torch.tensor(rng.choice(it[n], NB), device=dev)])[:, None]
            s = valeur(h, z, [[base_] + [1] * (n - base_)] * NB, rng)
            tn = np.full(NB, n)
            e_, _ = comparer_deux_ordres(m, zi, s, tn, tn)
            r["succ_chaine"][str(n)] = {"img_somme_egal": e_, "ecart_lignes_pas": float((s.mean().item() - zi.mean().item()) / k)}
    if h is not None and oubli:   # sum vs sum (terms <= 20), as in part A: exact "=", exact +/-1, and side
        r["oubli"] = {}
        for lo, hi in [(0, 20), (21, 40), (41, 60), (61, 100)]:
            G, D = [], []
            while len(G) < 3000:
                Tg = int(rng.integers(lo, hi + 1)); Td = Tg + int(rng.choice([0, 0, -1, 1, -2, 2, -5, 5]))
                if Td < 0:
                    continue
                if hi <= 60:
                    Lg = Ld = 3
                else:
                    Lg = max(int(rng.integers(4, 6)), -(-Tg // 20)); Ld = max(int(rng.integers(4, 6)), -(-Td // 20))
                if Td > 20 * Ld:
                    continue
                G.append(decomposer_sur(Tg, Lg, rng)); D.append(decomposer_sur(Td, Ld, rng))
            l = m.comparer(valeur(h, z, G, rng), valeur(h, z, D, rng)).float()
            tg = np.array([sum(x) for x in G]); td = np.array([sum(x) for x in D]); dd = np.abs(tg - td)
            p = l.argmax(1).cpu().numpy(); cote = ((l[:, GAUCHE] > l[:, DROITE]).cpu().numpy() == (tg > td))
            r["oubli"][f"{lo}-{hi}"] = {"egal": float((p[dd == 0] == EGAL).mean()),
                                        "ecart1_exact": float((p[dd == 1] == cible(tg[dd == 1], td[dd == 1])).mean()),
                                        "ecart1_cote": float(cote[dd == 1].mean())}
    return r


def charger(chemin_m, chemin_h):
    m = Comparateur(8, 1, 3, 128).to(dev); m.load_state_dict(torch.load(chemin_m, map_location=dev)); m.eval()
    h = None
    if chemin_h:
        h = Brique(n_ops=1).to(dev); h.load_state_dict(torch.load(chemin_h, map_location=dev)); h.eval()
    return m, h


def main():
    res = {}
    for f in sorted(glob.glob("results/au_dela_20_B/" + (sys.argv[1] if len(sys.argv) > 1 else "D*_seed*") + ".pt")):
        nom = os.path.basename(f)[:-3]
        meta = json.load(open(f[:-3] + ".json"))
        dep = meta["modele_depart"]
        ch = dep[:-3] + "_h.pt" if meta["type"] in ("addition", "succ") else None
        m, h = charger(f, ch)
        res[nom] = {"type": meta["type"], "depart": meta["depart"], "seed": meta["seed"], "apres": evaluer(m, h, np.random.default_rng(7), succ=meta["type"] == "succ", oubli=h is not None)}
        if h is not None:   # expression model before fine-tuning (same brick)
            m0, _ = charger(dep, None)
            res[nom]["avant"] = evaluer(m0, h, np.random.default_rng(7), succ=meta["type"] == "succ", oubli=True)
        print("fait", nom, flush=True)
    sortie = "" if len(sys.argv) == 1 else "_essai"
    json.dump(res, open(f"results/au_dela_20_test1{sortie}.json", "w"), indent=1)

    # table: means over the 3 seeds, per (starting model, type), at n = 21, 25, 30 and mean over 21..30
    L = []
    moy = lambda xs: np.mean(xs) if xs else float("nan")
    cles = [("lecture", "lecture"), ("img_img_egal", "img=img"), ("img_img_n_n1_cote", "img n/n+1 côté"),
            ("img_somme_egal", "img=somme"), ("img_somme_pm1_cote", "img/somme±1 côté"), ("img_somme_pm1_exact", "img/somme±1 exact"),
            ("ecart_lignes_pas", "écart lignes (pas)")]
    for N in [3, 9]:
        for typ in ["addition", "succ", "controle", "base"]:
            for quand in ["avant", "apres"]:
                rs = [v[quand] for v in res.values() if v["depart"] == N and v["type"] == typ and quand in v]
                if not rs:
                    continue
                L.append(f"départ 0-{N} {typ:8s} {quand:5s} ({len(rs)} graines)")
                for c, lab in cles:
                    vals = {n: [r["par_n"][str(n)][c] for r in rs if c in r["par_n"][str(n)]] for n in NS}
                    if not any(vals.values()):
                        continue
                    tout = [x for n in NS for x in vals[n]]
                    L.append(f"    {lab:20s} n=21 {moy(vals[21]):6.2f} | n=25 {moy(vals[25]):6.2f} | n=30 {moy(vals[30]):6.2f} | moy 21-30 {moy(tout):6.2f}")
                dp = [r["dans_plage"] for r in rs if "dans_plage" in r]
                if dp:
                    L.append("    dans la plage (img=somme / écart) " + " | ".join(f"n={n} {moy([x[str(n)]['img_somme_egal'] for x in dp]):.2f} / {moy([x[str(n)]['ecart_lignes_pas'] for x in dp]):+.1f}" for n in (10, 15, 20)))
    txt = "\n".join(L); print(txt); open(f"logs/au_dela_20_test1{sortie}.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    main()
