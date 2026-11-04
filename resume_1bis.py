"""Table for test 1bis (mixed fine-tuning) compared with part B (encoder only, h frozen) and with the control.

Sources: results/au_dela_20_1bis/*.json (before / after); results/au_dela_20_test1.json (part B and control, same measures).
Mean [min-max] over the 3 seeds. Usage (from the repository root): uv run python resume_1bis.py -> logs/resume_1bis.txt
"""
import glob, json
import numpy as np

r1 = {json.load(open(f))["nom"]: json.load(open(f)) for f in glob.glob("results/au_dela_20_1bis/D*.json")}
tb = json.load(open("results/au_dela_20_test1.json"))
f = lambda xs: (f"{np.mean(xs):.2f} [{min(xs):.2f}-{max(xs):.2f}]" if xs else "—")
NS = (21, 25, 28, 30)
L = []


def bloc(titre, rs):
    """rs: list of measurement dicts (output of au_dela_20_test1.evaluer)."""
    if not rs:
        return
    L.append(titre)
    g = lambda c, n: [r["par_n"][str(n)][c] for r in rs if c in r["par_n"][str(n)]]
    L.append("    lecture image       " + " | ".join(f"n={n} {f(g('lecture', n))}" for n in NS))
    if g("img_somme_egal", 21):
        L.append("    lecture somme       " + " | ".join(f"n={n} {f(g('lecture_somme', n))}" for n in NS))
        L.append("    image = somme       " + " | ".join(f"n={n} {f(g('img_somme_egal', n))}" for n in NS))
        L.append("    ±1 bon côté         " + " | ".join(f"n={n} {f(g('img_somme_pm1_cote', n))}" for n in NS))
        L.append("    écart lignes (pas)  " + " | ".join(f"n={n} {f(g('ecart_lignes_pas', n))}" for n in NS))
        dp = [r["dans_plage"] for r in rs]
        L.append("    dans la plage =     " + " | ".join(f"n={n} {f([x[str(n)]['img_somme_egal'] for x in dp])}" for n in (10, 15, 20)))
    L.append("    image n/n+1 côté    " + " | ".join(f"n={n} {f(g('img_img_n_n1_cote', n))}" for n in (21, 25, 28, 29)))
    if all("succ_chaine" in r for r in rs):
        sc = [r["succ_chaine"] for r in rs]
        L.append("    image = [20]+[1]×k  " + " | ".join(f"n={n} {f([x[str(n)]['img_somme_egal'] for x in sc])}" for n in (21, 25, 28, 30))
                 + " ; dans la plage [10]+[1]×k : " + " | ".join(f"n={n} {f([x[str(n)]['img_somme_egal'] for x in sc])}" for n in (15, 20)))
    if all("oubli" in r for r in rs):
        ob = [r["oubli"] for r in rs]
        L.append("    somme/somme =       " + " | ".join(f"{b} {f([x[b]['egal'] for x in ob])}" for b in ("0-20", "21-40", "41-60", "61-100")))
        L.append("    somme/somme ±1 côté " + " | ".join(f"{b} {f([x[b]['ecart1_cote'] for x in ob])}" for b in ("0-20", "21-40", "41-60", "61-100")))


for N in (3, 9):
    for v, typB in (("general", "addition"), ("succ", "succ")):
        L.append(f"\n===== départ 0-{N}, variante {v} =====")
        rs = [r for r in r1.values() if r["depart"] == N and r["variante"] == v]
        if rs:
            pas = [r["premier_pas_ok"] for r in rs]
            L.append(f"  1bis : premier pas où le critère de lecture est rempli : {pas} (plafond 6000)")
        bloc("  modèle d'expressions de départ (avant réglage)", [r["avant"] for r in rs])
        bloc("  1bis : réglage MÉLANGÉ, encodeur + tête + h", [r["apres"] for r in rs])
        bloc("  partie B : réglage images seules, h figé", [x["apres"] for x in tb.values() if x["depart"] == N and x["type"] == typB])
    bloc(f"\n===== départ 0-{N}, contrôle (comparaison seule, pas de h) =====", [x["apres"] for x in tb.values() if x["depart"] == N and x["type"] == "controle"])
txt = "\n".join(L); print(txt); open("logs/resume_1bis.txt", "w").write(txt + "\n")
