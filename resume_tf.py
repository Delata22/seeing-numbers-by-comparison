"""Transformer head versus MLP head (ranges 0-3, 0-5, 0-9 with "equal"). Usage (from the repository root): uv run python resume_tf.py -> logs/plage/resume_tf.txt"""
import glob, json
from collections import defaultdict
import numpy as np
G = defaultdict(list)
for f in glob.glob("results/plage/N0-*_eg_seed*/metrics.json"):
    r = json.load(open(f))
    if r["largeur"] != 8 or r["dim_z"] != 1:
        continue
    G[(r["n_max"], r.get("tete_type", "mlp"))].append(r)
mm = lambda xs, f="{:.2f}": (f.format(np.mean(xs)) + " [" + f.format(min(xs)) + "-" + f.format(max(xs)) + "]") if xs else "—"
L = ["plage tête                    (n) params tête  0-N   R²    | portée taille     lit 10 / 15 / 20     côté N+1..2N   côté >2N      "
     "« égal » n/n+1 au-delà  « égal » n=n au-delà  symétrie dans / au-delà | portée aire  arrêt"]
for (N, t), rs in sorted(G.items()):
    tg = lambda r: r["au_dela"]["arret"]["taille"]
    lec = " / ".join(f"{np.mean([tg(r)['lecture'][str(n)] for r in rs]):.1f}" for n in (10, 15, 20))
    eg = [np.mean([v["egal"] for k, v in tg(r)["voisins"].items() if int(k) > N]) for r in rs]
    egnn = [np.mean([v for k, v in tg(r)["egal_n_n"].items() if int(k) > N]) for r in rs]
    sym = lambda k: [tg(r)["symetrie"][k] for r in rs if "symetrie" in tg(r)]
    L.append(f"0-{N:<2d}  {t:22s} ({len(rs)}) {rs[0]['params_tete']:6d}   {np.mean([r['dans_1_N']['inegales'] for r in rs]):.3f} "
             f"{np.mean([r['dans_1_N']['sonde_r2'] for r in rs]):.3f} | {mm([tg(r)['portee'] for r in rs], '{:.1f}'):16s} {lec:20s} "
             f"{mm([tg(r)['bandes'].get('N+1_2N', {}).get('bon_cote', np.nan) for r in rs]):14s} "
             f"{mm([tg(r)['bandes'].get('au_dela_2N', {}).get('bon_cote', np.nan) for r in rs]):14s} {mm(eg):22s} {mm(egnn):20s} "
             f"{mm(sym('dans_plage'))} / {mm(sym('au_dela'))} | {mm([r['au_dela']['arret']['aire']['portee'] for r in rs], '{:.1f}'):12s} "
             f"{np.mean([r['pas_arret'] for r in rs]):.0f}")
L.append("\nBon côté n contre n+1 (taille fixe), n = 1..19 :")
for (N, t), rs in sorted(G.items()):
    L.append(f"  0-{N:<2d} {t:22s}: " + " ".join(f"{np.mean([r['au_dela']['arret']['taille']['voisins'][str(n)]['bon_cote'] for r in rs]):.2f}" for n in range(1, 20)))
txt = "\n".join(L); print(txt); open("logs/plage/resume_tf.txt", "w").write(txt + "\n")
