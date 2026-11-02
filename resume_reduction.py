"""Table for the shrinking experiment: one row per (width, dim_z, with/without equality), mean [min-max] over seeds.

Usage (from the repository root): uv run python resume_reduction.py  -> screen + logs/reduction/resume.txt
"""
import glob, json
from collections import defaultdict
import numpy as np

groupes = defaultdict(list)
for f in glob.glob("results/reduction/*/metrics.json"):
    r = json.load(open(f))
    groupes[(r["sans_egal"], r["dim_z"], r["largeur"], r.get("tete", 128))].append(r)


def mm(xs, fmt="{:.3f}"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    return (fmt.format(np.mean(xs)) + " [" + fmt.format(min(xs)) + "-" + fmt.format(max(xs)) + "]")


lignes = []
entete = ("égal  z   larg  tête  params enc/total   (n)  inégales 1-9            retenues   transit.  sonde R²  "
          "axe1 ACP  pas arrêt         | taille 10-20 bon côté   « égal » n/n+1     lit 20            "
          "| aire 10-30 bon côté     lit 30")
lignes.append(entete)
for (se, z, L, T), rs in sorted(groupes.items()):
    R = lambda k: [r["resume"][k] for r in rs]
    G = lambda mode, k: [r["generalisation"]["arret"][mode][k] for r in rs]
    lect = lambda mode, n: [r["generalisation"]["arret"][mode]["lecture"][str(n)] for r in rs]
    lignes.append(
        f"{'sans' if se else 'avec'}  {z:2d}  1/{L:<2d}  {T:3d}   {rs[0]['params_encodeur']:6d}/{rs[0]['params_encodeur'] + rs[0]['params_tete']:<6d}   ({len(rs)})  "
        f"{mm(R('inegales'))}  {np.mean(R('retenues')):.3f}     {np.mean(R('transitivite')):.3f}     "
        f"{np.mean(R('sonde_r2')):.3f}     {np.mean(R('acp_axe1')):.2f}     "
        f"{mm([r['pas_arret'] for r in rs], '{:.0f}'):16s} | {mm(G('taille', 'bon_cote'))}  "
        f"{mm(G('taille', 'egal_n_n1'), '{:.2f}'):17s}  {mm(lect('taille', 20), '{:.1f}'):16s}  | "
        f"{mm(G('aire', 'bon_cote'))}  {mm(lect('aire', 30), '{:.1f}')}")

# Generalization over training (mean over seeds and width conditions)
lignes.append("\nGénéralisation selon le pas (taille 10-20 bon côté / « égal » n/n+1 / aire 10-30 bon côté), z = 16 :")
for se in [False, True]:
    for L in sorted({k[2] for k in groupes if k[1] == 16}):
        rs = groupes.get((se, 16, L, 128), [])
        if not rs:
            continue
        cases = []
        for p in ["200", "500", "1000", "arret"]:
            gs = [r["generalisation"][p] for r in rs if p in r["generalisation"]]
            if gs:
                cases.append(f"{p}: {np.mean([g['taille']['bon_cote'] for g in gs]):.3f}/"
                             f"{np.mean([g['taille']['egal_n_n1'] for g in gs]):.2f}/"
                             f"{np.mean([g['aire']['bon_cote'] for g in gs]):.3f}")
        lignes.append(f"  {'sans' if se else 'avec'} égal, 1/{L:<2d}  " + "   ".join(cases))

# Failed (unequal) pairs on 1-9 for models below 100 %
lignes.append("\nPaires inégales ratées sur 1-9 (exactitude < 0,99) :")
for (se, z, L, T), rs in sorted(groupes.items()):
    for r in rs:
        rates = [f"{k} {v['acc']:.2f} (égal {v['repart'][1]:.2f}, mauvais côté {1 - v['acc'] - v['repart'][1]:.2f})" for k, v in r["complet"]["mix"]["paires"].items()
                 if k.split("-")[0] != k.split("-")[1] and v["acc"] < 0.99]
        if rates:
            lignes.append(f"  {r['nom']}: " + ", ".join(rates))

texte = "\n".join(lignes)
print(texte)
open("logs/reduction/resume.txt", "w").write(texte + "\n")
