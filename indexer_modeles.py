"""Collects every trained model into modeles/<config>/, with metadata and an index.

Later experiments (addition, expressions, analyses) load their starting models from there.
The models are small, so ALL seeds are copied.

For each configuration: modeles/<config>/seed<k>.pt + seed<k>.json (architecture, condition,
measurements, reload command). Best seed = highest accuracy on the unequal pairs
from 1 to 9; on a tie, lowest validation loss (series with early stopping) or best
"correct side" at fixed size 10-20 (older series, if measured).
Index: modeles/INDEX.md. Re-run after each series: the script only copies what is missing.

Usage (from the repository root): uv run python indexer_modeles.py
"""
import glob, json, os, shutil
import numpy as np

os.makedirs("modeles", exist_ok=True)
configs = {}


def inegales(paires):
    return float(np.mean([v["acc"] for k, v in paires.items() if k.split("-")[0] != k.split("-")[1]]))


# "Comparison 1-9" series (compare.py): encoder 32-32-64-64-128-128, z = 64, head 128
for f in sorted(glob.glob("results/*_seed[0-9]/metrics.json")):
    run = f.split("/")[1]
    v, c, s = run.split("_")
    if v not in ("mix", "uni"):
        continue
    r = json.load(open(f))
    g = f"results/grands/{run}.json"
    cote = None
    if os.path.exists(g):
        gj = json.load(open(g))
        try:
            M = np.array(gj["taille"]["ordre"], dtype=float)
            cote = float(np.nanmean([M[a, b] for a in range(9, M.shape[0]) for b in range(9, M.shape[0]) if a != b]))
        except (KeyError, IndexError):
            pass
    sans_egal = c.endswith("se")
    configs.setdefault(f"comparaison_{v}_{c}", []).append({
        "source": f"results/{run}/modele.pt", "seed": int(s[4:]),
        "score": (inegales(r["entraine"][v]["paires"]), cote if cote is not None else 0.0),
        "meta": {"serie": "comparaison 1-9 (compare.py)", "version": v, "condition": c,
                 "architecture": "encodeur run.Encodeur 32-32-64-64-128-128, z = 64, tête 128 neurones",
                 "classe_egal": not sans_egal,
                 "paires_retenues": "3-6, 5-8, 2-7 (deux sens)" + (" + 6=6" if c.startswith("B") else ""),
                 "pas": 4000, "inegales_1_9": inegales(r["entraine"][v]["paires"]),
                 "bon_cote_taille_10_20": cote,
                 "recharger": "from compare import charger_modele; m = charger_modele(chemin, 'cuda')"}})

# "Shrinking" series (reduction.py)
for f in sorted(glob.glob("results/reduction/*/metrics.json")):
    r = json.load(open(f))
    t = r.get("tete", 128)
    eg = "sans_egal" if r["sans_egal"] else "egal"
    cfg = f"reduction_x1-{r['largeur']}_z{r['dim_z']}_tete{t}_{eg}"
    perte = min(p["perte_val"] for p in r["suivi"])
    gt = r["generalisation"]["arret"]["taille"]
    configs.setdefault(cfg, []).append({
        "source": f"results/reduction/{r['nom']}/modele.pt", "seed": r["seed"],
        "score": (r["resume"]["inegales"], -perte),
        "meta": {"serie": "réduction (reduction.py)", "largeur": f"x1/{r['largeur']}",
                 "canaux": [max(1, x // r["largeur"]) for x in (32, 32, 64, 64, 128, 128)],
                 "dim_z": r["dim_z"], "tete": t, "classe_egal": not r["sans_egal"],
                 "paires_retenues": "3-6, 5-8, 2-7 (deux sens)",
                 "params_encodeur": r["params_encodeur"], "params_total": r["params_encodeur"] + r["params_tete"],
                 "pas_arret": r["pas_arret"], "perte_val_min": perte,
                 "inegales_1_9": r["resume"]["inegales"], "sonde_r2": r["resume"]["sonde_r2"],
                 "bon_cote_taille_10_20": gt["bon_cote"], "egal_n_n1_taille": gt["egal_n_n1"],
                 "bon_cote_aire_10_30": r["generalisation"]["arret"]["aire"]["bon_cote"],
                 "recharger": (f"from reduction import Comparateur; import torch; "
                               f"m = Comparateur({r['largeur']}, {r['dim_z']}, {2 if r['sans_egal'] else 3}, {t}); "
                               f"m.load_state_dict(torch.load(chemin))")}})

# "Range 1 to N" series (plage.py)
for f in sorted(glob.glob("results/plage/*/metrics.json")):
    r = json.load(open(f))
    eg = "sans_egal" if r["sans_egal"] else "egal"
    tt = r.get('tete_type', 'mlp'); tnom = f"tete{r['tete']}" if tt == 'mlp' else tt
    cfg = f"plage_N{'0-' if r.get('zero') else ''}{r['n_max']}_x1-{r['largeur']}_z{r['dim_z']}_{tnom}_{eg}"
    perte = min(p["perte_val"] for p in r["suivi"])
    gt = r["au_dela"]["arret"]["taille"]
    configs.setdefault(cfg, []).append({
        "source": f"results/plage/{r['nom']}/modele.pt", "seed": r["seed"],
        "score": (r["dans_1_N"]["inegales"], -perte),
        "meta": {"serie": "plage 1 à N (plage.py)", "n_max": r["n_max"], "largeur": f"x1/{r['largeur']}",
                 "canaux": [max(1, x // r["largeur"]) for x in (32, 32, 64, 64, 128, 128)],
                 "dim_z": r["dim_z"], "tete": r["tete"], "classe_egal": not r["sans_egal"],
                 "paires_retenues": "aucune",
                 "params_encodeur": r["params_encodeur"], "params_total": r["params_encodeur"] + r["params_tete"],
                 "pas_arret": r["pas_arret"], "perte_val_min": perte,
                 "inegales_1_9": r["dans_1_N"]["inegales"], "sonde_r2": r["dans_1_N"]["sonde_r2"],
                 "bon_cote_taille_10_20": gt["bandes"].get("N+1_2N", {}).get("bon_cote"),
                 "portee_taille": gt["portee"],
                 "recharger": (f"from reduction import Comparateur; import torch; "
                               f"m = Comparateur({r['largeur']}, {r['dim_z']}, {2 if r['sans_egal'] else 3}, {r['tete']}); "
                               f"m.load_state_dict(torch.load(chemin))")}})

# 1-100 series (cent.py): 192x192, reduction.Encodeur encoder with a 12x12 projection
for f in sorted(glob.glob("results/cent/*/metrics.json")):
    r = json.load(open(f))
    eg = "sans_egal" if r["sans_egal"] else "egal"
    cfg = f"cent_{r['distrib']}_x1-{r['largeur']}_z{r['dim_z']}_tete{r['tete']}_{eg}"
    mz = r["mesures"]
    configs.setdefault(cfg, []).append({
        "source": f"results/cent/{r['nom']}/modele.pt", "seed": r["seed"],
        "score": (np.mean([mz[b]["exact_inegales"] for b in ["1-9", "10-20", "21-40", "41-70", "71-100"]]), -r["perte_val_min"]),
        "meta": {"serie": "1-100 (cent.py)", "distrib": r["distrib"], "largeur": f"x1/{r['largeur']}",
                 "image": "192x192, rayon 2,5-4,5 px", "dim_z": r["dim_z"], "tete": r["tete"],
                 "classe_egal": not r["sans_egal"], "pas_arret": r["pas_arret"], "interrompu": r.get("interrompu", False),
                 "inegales_1_9": mz["1-9"]["exact_inegales"],
                 "bon_cote_taille_10_20": mz["10-20"]["bon_cote"],
                 "exact_par_bande": {b: mz[b]["exact_inegales"] for b in ["1-9", "10-20", "21-40", "41-70", "71-100"]},
                 "recharger": (f"from reduction import Comparateur; import torch, torch.nn as nn; "
                               f"m = Comparateur({r['largeur']}, {r['dim_z']}, {2 if r['sans_egal'] else 3}, {r['tete']}); "
                               f"m.enc.proj = nn.Linear(m.enc.proj.in_features // 16 * 144, {r['dim_z']}); "
                               f"m.load_state_dict(torch.load(chemin))")}})

# Addition series (addition.py): encoder + comparison head (modele.pt) and addition module h (h.pt)
for f in sorted(glob.glob("results/addition/*/metrics.json")):
    r = json.load(open(f))
    cfg = f"addition_depart0-{r['depart']}_{r['variante']}"
    ad = r["apres"]["addition"]
    sc = float(np.mean([v["c=s"] for v in ad.values()]))
    d = f"modeles/{cfg}"; os.makedirs(d, exist_ok=True)
    hsrc = f"results/addition/{r['nom']}/h.pt"
    if os.path.exists(hsrc):
        shutil.copy2(hsrc, f"{d}/seed{r['seed']}_h.pt")
    configs.setdefault(cfg, []).append({
        "source": f"results/addition/{r['nom']}/modele.pt", "seed": r["seed"], "score": (sc, 0.0),
        "meta": {"serie": "addition (addition.py)", "depart": r["modele_depart"], "variante": r["variante"],
                 "pas_arret": r["pas_arret"], "A_final": r["A_final"], "inegales_1_9": r["apres"]["comparaison_0_N"],
                 "bon_cote_taille_10_20": None, "addition_c_egal_s_par_bande": {k: v["c=s"] for k, v in ad.items()},
                 "fichier_h": f"{d}/seed{r['seed']}_h.pt",
                 "recharger": ("from reduction import Comparateur; from addition import Additionneur; import torch; "
                               "m = Comparateur(8, 1, 3, 128); m.load_state_dict(torch.load(chemin)); "
                               "h = Additionneur(); h.load_state_dict(torch.load(fichier_h))")}})

# Expression series (expressions.py: "+"; expressions_moins.py: "+" and "-"): modele.pt + brick h.pt (h and symbols)
for f in sorted(glob.glob("results/expressions*/*/metrics.json")):
    r = json.load(open(f))
    serie = f.split("/")[1]
    cfg = f"{serie}_depart0-{r['depart']}_{r['variante']}"
    a = r["apres"]
    sc = float(np.mean([v.get("egal", {}).get("exact", 0) for v in a.get("longueurs", {}).values()] or [0]))
    d = f"modeles/{cfg}"; os.makedirs(d, exist_ok=True)
    hsrc = os.path.join(os.path.dirname(f), "h.pt")
    if os.path.exists(hsrc):
        shutil.copy2(hsrc, f"{d}/seed{r['seed']}_h.pt")
    configs.setdefault(cfg, []).append({
        "source": os.path.join(os.path.dirname(f), "modele.pt"), "seed": r["seed"], "score": (sc, 0.0),
        "meta": {"serie": serie, "depart": r["modele_depart"], "variante": r["variante"], "pas_arret": r["pas_arret"],
                 "A_final": r["A_final"], "inegales_1_9": a["comparaison_0_N"], "bon_cote_taille_10_20": None,
                 "fichier_h": f"{d}/seed{r['seed']}_h.pt",
                 "recharger": ("from reduction import Comparateur; from expressions import Brique; import torch; "
                               "m = Comparateur(8, 1, 3, 128); m.load_state_dict(torch.load(chemin)); "
                               f"h = Brique(n_ops={r.get('n_ops', 1)}); h.load_state_dict(torch.load(fichier_h))")}})

# Beyond 20, part B and control (au_dela_20_B.py): comparison model after fine-tuning on images 0..30.
# Score: min correct side n versus n+1 (21..29) at the end, then lowest max readout error.
# The brick h (addition, succ) is not fine-tuned: it is the one from the starting expressions model (fichier_h_non_regle).
for f in sorted(glob.glob("results/au_dela_20_B/D*_seed*.json")):
    r = json.load(open(f))
    src = f[:-5] + ".pt"
    if not os.path.exists(src):
        continue
    fin = r["suivi"][max(r["suivi"], key=int)]
    cfg = f"au_dela_20_B_depart0-{r['depart']}_{r['type']}"
    h = r["modele_depart"][:-3] + "_h.pt" if r["type"] in ("addition", "succ") else None
    configs.setdefault(cfg, []).append({
        "source": src, "seed": r["seed"], "score": (fin["cote_min"], -fin["ecart_lecture_max"]),
        "meta": {"serie": "au-delà de 20, partie B (au_dela_20_B.py)", "depart": r["modele_depart"], "type": r["type"],
                 "pas_controle": r["pas_controle"], "premier_pas_ok": r["premier_pas_ok"], "pas_arret": r["pas_arret"],
                 "inegales_1_9": fin["cote_min"], "bon_cote_taille_10_20": None,
                 "ecart_lecture_max_21_30": fin["ecart_lecture_max"], "fichier_h_non_regle": h,
                 "recharger": ("from reduction import Comparateur; import torch; "
                               "m = Comparateur(8, 1, 3, 128); m.load_state_dict(torch.load(chemin))")}})

# Beyond 20, test 1bis and long follow-up (au_dela_20_1bis.py): mixed fine-tuning, model + brick h tuned together.
# Score: mean image = sum on n = 21..30 after fine-tuning (succ variant: image = [20] + [1] x k), then min side n / n+1.
for serie in ("au_dela_20_1bis", "au_dela_20_1bis_long"):
    for f in sorted(glob.glob(f"results/{serie}/D*_seed*.json")):
        r = json.load(open(f))
        src = f[:-5] + ".pt"
        if not os.path.exists(src):
            continue
        a = r["apres"]
        if r["variante"] == "succ":
            eg = float(np.mean([a["succ_chaine"][str(n)]["img_somme_egal"] for n in range(21, 31)]))
        else:
            eg = float(np.mean([a["par_n"][str(n)]["img_somme_egal"] for n in range(21, 31)]))
        cote = float(min(a["par_n"][str(n)]["img_img_n_n1_cote"] for n in range(21, 30)))
        cfg = f"{serie}_depart0-{r['depart']}_{r['variante']}"
        d = f"modeles/{cfg}"; os.makedirs(d, exist_ok=True)
        shutil.copy2(f[:-5] + "_h.pt", f"{d}/seed{r['seed']}_h.pt")
        configs.setdefault(cfg, []).append({
            "source": src, "seed": r["seed"], "score": (eg, cote),
            "meta": {"serie": f"au-delà de 20, {serie} (au_dela_20_1bis.py)", "depart": r["modele_depart"], "variante": r["variante"],
                     "premier_pas_ok": r["premier_pas_ok"], "pas_arret": r["pas_arret"], "inegales_1_9": cote,
                     "bon_cote_taille_10_20": None, "image_egal_somme_21_30": eg, "fichier_h": f"{d}/seed{r['seed']}_h.pt",
                     "recharger": ("from reduction import Comparateur; from expressions import Brique; import torch; "
                                   "m = Comparateur(8, 1, 3, 128); m.load_state_dict(torch.load(chemin)); "
                                   "h = Brique(n_ops=1); h.load_state_dict(torch.load(fichier_h))")}})

# "Next word" project (mot_suivant/mot_suivant.py): full model (encoder, memory, notebook, outputs)
for f in sorted(glob.glob("mot_suivant/results/*/metrics.json")):
    r = json.load(open(f))
    cfg = f"mot_suivant_{r['ordre']}_{r['rythme']}_{r['cible']}_W{r['W']}_{'carnet' if r['carnet'] else 'sans'}"
    au = r["final"]["autonomie"]
    sc = float(np.mean([au[str(n)]["parfait"] for n in range(1, 11)]))
    configs.setdefault(cfg, []).append({
        "source": os.path.join(os.path.dirname(f), "modele.pt"), "seed": r["seed"], "score": (sc, 0.0),
        "meta": {"serie": "mot suivant, vague 1", "ordre": r["ordre"], "rythme": r["rythme"], "cible": r["cible"], "W": r["W"],
                 "carnet": r["carnet"], "pas": r["pas"], "inegales_1_9": sc, "bon_cote_taille_10_20": None,
                 "autonomie_parfaite_1_10": sc,
                 "recharger": ("import sys, torch; sys.path.insert(0, 'mot_suivant'); from mot_suivant import Modele; "
                               f"m = Modele({r['W']}, {r['carnet']}); m.load_state_dict(torch.load(chemin))")}})

lignes = ["# Modèles sauvegardés\n",
          "Une ligne par configuration. Toutes les graines sont copiées ; ★ = meilleure graine "
          "(exactitude 1-9 sur paires inégales, puis perte de validation ou bon côté 10-20).\n",
          "| configuration | graines | ★ | inégales 1-9 ou 1-N (★) | bon côté taille 10-20 ou N+1..2N (★) | paramètres | fichier ★ |",
          "|---|---|---|---|---|---|---|"]
for cfg, rs in sorted(configs.items()):
    d = f"modeles/{cfg}"
    os.makedirs(d, exist_ok=True)
    best = max(rs, key=lambda x: x["score"])
    for x in rs:
        dst = f"{d}/seed{x['seed']}.pt"
        if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(x["source"]):
            shutil.copy2(x["source"], dst)
        meta = dict(x["meta"], seed=x["seed"], meilleure_graine=x is best, chemin=dst)
        json.dump(meta, open(f"{d}/seed{x['seed']}.json", "w"), indent=1, ensure_ascii=False)
    m = best["meta"]
    cote = m.get("bon_cote_taille_10_20")
    lignes.append(f"| {cfg} | {len(rs)} | {best['seed']} | {m['inegales_1_9']:.3f} | "
                  f"{'—' if cote is None else f'{cote:.3f}'} | {m.get('params_total', '~430 000')} | "
                  f"{d}/seed{best['seed']}.pt |")
open("modeles/INDEX.md", "w").write("\n".join(lignes) + "\n")
print(f"{len(configs)} configurations, {sum(len(v) for v in configs.values())} modèles -> modeles/INDEX.md")
