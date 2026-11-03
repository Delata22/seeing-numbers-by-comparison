"""Comparison table for the 1-100 series: one row per condition (distribution x with/without "equal"),
mean [min-max] over seeds. Runs without "equal" and without the fix (gradient clipping; unstable) are listed separately.

Measures (fresh images, uniform 1-100, 40 per number):
  - exact: correct 3-class answer on unequal pairs (a model with "equal" loses points
    when it answers "equal"; a model without "equal" cannot);
  - side: the score leans to the correct side (comparable between the two families);
  - side n/n+1: same for neighbouring pairs;
  - "equal" n/n+1 and n = n: models with "equal".
Usage (from the repository root): uv run python resume_cent.py -> screen + logs/cent/resume.txt
"""
import glob, json
from collections import defaultdict
import numpy as np

BANDES = ["1-9", "10-20", "21-40", "41-70", "71-100"]
G = defaultdict(list)
for f in glob.glob("results/cent/*/metrics.json"):
    r = json.load(open(f))
    if r["sans_egal"]:
        fam = "sans égal" if r.get("clip", 0) > 0 else "sans égal, sans correctif"
    else:
        fam = "avec égal"
    G[(fam, r["distrib"])].append(r)


def mm(xs, fmt="{:.3f}"):
    return fmt.format(np.mean(xs)) + " [" + fmt.format(min(xs)) + "-" + fmt.format(max(xs)) + "]"


L = []
for titre, cle, fmt in [("EXACT (paires inégales)", "exact_inegales", "{:.3f}"), ("CÔTÉ (toutes paires inégales)", "bon_cote", "{:.3f}"),
                        ("CÔTÉ n contre n+1", "bon_cote_n_n1", "{:.2f}"), ("« ÉGAL » sur n contre n+1", "egal_n_n1", "{:.2f}"),
                        ("« ÉGAL » sur n contre n+2", "egal_n_n2", "{:.2f}"), ("« ÉGAL » sur n = n", "egal_n_n", "{:.2f}")]:
    L.append(f"\n{titre}")
    L.append("famille                     distrib    (n)  " + "  ".join(f"{b:>20s}" for b in BANDES))
    for (fam, d), rs in sorted(G.items()):
        if "égal »" in titre and fam != "avec égal":
            continue
        L.append(f"{fam:27s} {d:9s} ({len(rs)})  " + "  ".join(f"{mm([r['mesures'][b][cle] for r in rs], fmt):>20s}" for b in BANDES))

L.append("\nPAR RAPPORT min/max (exact) et LIGNE DANS z")
for (fam, d), rs in sorted(G.items()):
    rap = "  ".join(f"{k}: {np.mean([r['mesures']['rapport'][k]['exact'] for r in rs]):.3f}" for k in ["<=0.8", "0.8-0.9", "0.9-0.95", ">0.95"])
    li = lambda k: np.mean([r["mesures"]["ligne"][k] for r in rs])
    L.append(f"{fam:27s} {d:9s} | {rap} | R² lin {li('r2_lineaire'):.3f} log {li('r2_log'):.3f} | étalement ∝ n^{li('pente_etalement_loglog'):.2f}"
             f" | pas arrêt {mm([r['pas_arret'] for r in rs], '{:.0f}')}")
txt = "\n".join(L)
print(txt)
open("logs/cent/resume.txt", "w").write(txt + "\n")
