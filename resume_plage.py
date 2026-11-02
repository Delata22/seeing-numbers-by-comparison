"""Table for the "range 1 to N" experiment: one row per (N, with/without equality), mean [min-max] over seeds.
Usage (from the repository root): uv run python resume_plage.py -> screen + logs/plage/resume.txt
"""
import glob, json
from collections import defaultdict
import numpy as np

G = defaultdict(list)
for f in glob.glob("results/plage/*/metrics.json"):
    r = json.load(open(f)); G[(r["sans_egal"], r["n_max"], r.get("zero", False))].append(r)

def mm(xs, fmt="{:.3f}"):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return "—" if not xs else fmt.format(np.mean(xs)) + " [" + fmt.format(min(xs)) + "-" + fmt.format(max(xs)) + "]"

L = ["égal  N   (n)  1-N inég.   R²     arrêt | TAILLE portée (×N)         N+1..2N bon côté      >2N bon côté          "
     "lit 20            « égal » n/n+1 (N+1..20) | AIRE portée   lit 30"]
for (se, N, z0), rs in sorted(G.items()):
    a = lambda r: r["au_dela"]["arret"]
    t = lambda r: a(r)["taille"]
    eg = lambda r: np.mean([v["egal"] for k, v in t(r)["voisins"].items() if int(k) > N]) if not se else None
    L.append(f"{'sans' if se else 'avec'}  {('0-' if z0 else '1-') + str(N):>4s} ({len(rs)})  {np.mean([r['dans_1_N']['inegales'] for r in rs]):.3f}   "
             f"{np.mean([r['dans_1_N']['sonde_r2'] for r in rs]):.3f}  {np.mean([r['pas_arret'] for r in rs]):5.0f} | "
             f"{mm([t(r)['portee'] for r in rs], '{:.1f}')} (×{np.mean([t(r)['facteur'] for r in rs]):.2f})   "
             f"{mm([t(r)['bandes'].get('N+1_2N', {}).get('bon_cote') for r in rs]):20s}  "
             f"{mm([t(r)['bandes'].get('au_dela_2N', {}).get('bon_cote') for r in rs]):20s}  "
             f"{mm([t(r)['lecture']['20'] for r in rs], '{:.1f}'):16s}  {mm([eg(r) for r in rs], '{:.2f}'):24s} | "
             f"{mm([a(r)['aire']['portee'] for r in rs], '{:.1f}'):14s} {mm([a(r)['aire']['lecture']['30'] for r in rs], '{:.1f}')}")

L.append("\nLecture de la droite (taille fixe), moyenne des graines, n = 1..20 :")
for (se, N, z0), rs in sorted(G.items()):
    lec = [np.mean([r["au_dela"]["arret"]["taille"]["lecture"][str(n)] for r in rs]) for n in range(1, 21)]
    L.append(f"  {'sans' if se else 'avec'} {('0-' if z0 else '1-') + str(N):>4s} : " + " ".join(f"{x:.1f}" for x in lec))
L.append("\nBon côté n contre n+1 (taille fixe), n = 1..19 :")
for (se, N, z0), rs in sorted(G.items()):
    bc = [np.mean([r["au_dela"]["arret"]["taille"]["voisins"][str(n)]["bon_cote"] for r in rs]) for n in range(1, 20)]
    L.append(f"  {'sans' if se else 'avec'} {('0-' if z0 else '1-') + str(N):>4s} : " + " ".join(f"{x:.2f}" for x in bc))
L.append("\n« égal » pour n contre n (deux images, taille fixe), n = 1..20, modèles avec égal :")
for (se, N, z0), rs in sorted(G.items()):
    if se: continue
    e = [np.mean([r["au_dela"]["arret"]["taille"]["egal_n_n"][str(n)] for r in rs]) for n in range(1, 21)]
    L.append(f"  {('0-' if z0 else '1-') + str(N):>4s} : " + " ".join(f"{x:.2f}" for x in e))
L.append("\nPortée en taille fixe selon le pas (200 / 500 / 1000 / arrêt) :")
for (se, N, z0), rs in sorted(G.items()):
    c = []
    for p in ["200", "500", "1000", "arret"]:
        v = [r["au_dela"][p]["taille"]["portee"] for r in rs if p in r["au_dela"]]
        if v: c.append(f"{p}: {np.mean(v):.1f}")
    L.append(f"  {'sans' if se else 'avec'} {('0-' if z0 else '1-') + str(N):>4s} : " + "   ".join(c))
txt = "\n".join(L); print(txt); open("logs/plage/resume.txt", "w").write(txt + "\n")
