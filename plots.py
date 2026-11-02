"""Plots for the "comparison only" experiment. Averaged over seeds.

Usage (from the repository root): uv run python plots.py [mix|uni]   -> figures/<version>/*.png
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.decomposition import PCA

D = sys.argv[1] if len(sys.argv) > 1 else "mix"   # version: mix or uni
FORMES = [D]
NOM = {"mix": "plusieurs formes par image", "uni": "une forme par image"}
COUL = {D: "#2a78d6"}
GRIS = "#8a8984"
COND = {"A": "A : 3-6, 5-8, 2-7 jamais vus", "B": "B : 3-6, 5-8, 2-7 et 6 = 6 jamais vus",
        "B2s": "B, chaque paire dans les deux sens", "Ase": "A sans classe « égal »"}
N = np.arange(1, 10)
FIG = f"figures/{D}"
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
                     "axes.edgecolor": "#8a8984", "figure.facecolor": "white"})


def runs(cond):
    return [json.load(open(f)) for f in sorted(glob.glob(f"results/{D}_{cond}_seed*/metrics.json"))]


R = {c: runs(c) for c in COND}
R = {c: r for c, r in R.items() if r}


def moy(liste):
    a = np.array(liste, dtype=float)
    return a.mean(0), a.std(0)


# 1. Comparison matrix: accuracy for each pair (left a, right b)
fig, axs = plt.subplots(1, len(R), figsize=(5.2 * len(R), 4.6), squeeze=False)
for ax, (c, rs) in zip(axs[0], R.items()):
    M = np.zeros((9, 9))
    for r in rs:
        for k, v in r["entraine"][D]["paires"].items():
            a, b = map(int, k.split("-"))
            M[a - 1, b - 1] += v["acc"] / len(rs)
    im = ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1, origin="lower")
    for a in range(9):
        for b in range(9):
            ax.text(b, a, f"{100 * M[a, b]:.0f}", ha="center", va="center", fontsize=7,
                    color="white" if M[a, b] > 0.8 else "#0b0b0b")
    p = rs[0]["entraine"][D]["paires"]
    for k, v in p.items():
        if v["retenue"]:
            a, b = map(int, k.split("-"))
            ax.add_patch(Rectangle((b - 1.5, a - 1.5), 1, 1, fill=False, ec="#e34948", lw=2))
    ax.set_xticks(range(9), N); ax.set_yticks(range(9), N); ax.grid(False)
    ax.set_xlabel("quantité à droite"); ax.set_ylabel("quantité à gauche")
    ax.set_title(f"{COND[c]}\n% de bonnes réponses (cadre rouge = jamais vu)", fontsize=10)
fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8)
fig.savefig(f"{FIG}/1_matrice_comparaison.png", dpi=130, bbox_inches="tight"); plt.close()

# 2. Weber's law: accuracy vs the min/max ratio, per shape (condition c0)
c0 = "B"
fig, ax = plt.subplots(figsize=(6.4, 4.2))
for f in FORMES:
    pts = {}
    for r in R[c0]:
        for k, v in r["entraine"][f]["paires"].items():
            a, b = map(int, k.split("-"))
            if a != b:
                pts.setdefault(round(v["ratio"], 3), []).append(v["acc"])
    xs = sorted(pts)
    ax.plot(xs, [np.mean(pts[x]) for x in xs], "-o", ms=4, lw=2, color=COUL[f], label=NOM[f])
ax.set_xlabel("rapport min/max des deux quantités (1 = presque égales)")
ax.set_ylabel("exactitude de comparaison")
ax.set_title(f"Loi de Weber ? (paires inégales, condition {c0}, moyenne sur graines)", fontsize=10)
ax.legend(frameon=False)
fig.savefig(f"{FIG}/2_weber.png", dpi=130, bbox_inches="tight"); plt.close()

# 3. Linear "how many?" probe per quantity: trained vs untrained, per shape
fig, axs = plt.subplots(1, len(R), figsize=(5.6 * len(R), 4.0), squeeze=False, sharey=True)
for ax, (c, rs) in zip(axs[0], R.items()):
    for f in FORMES:
        m, s = moy([[r["entraine"]["sondes"][f]["classif_par_n"][str(n)] for n in N] for r in rs])
        ax.plot(N, m, "-o", ms=4, lw=2, color=COUL[f], label=f"{NOM[f]} (entraîné)")
        ax.fill_between(N, m - s, m + s, color=COUL[f], alpha=0.15, lw=0)
        m0, _ = moy([[r["non_entraine"]["sondes"][f]["classif_par_n"][str(n)] for n in N] for r in rs])
        ax.plot(N, m0, "--", lw=1.2, color=COUL[f], alpha=0.6)
    ax.axhline(1 / 9, color=GRIS, lw=1, ls=":")
    ax.text(9.1, 1 / 9, "hasard", color=GRIS, va="center", fontsize=8)
    ax.set_xticks(N); ax.set_xlabel("quantité réelle"); ax.set_ylim(0, 1.02)
    ax.set_title(f"{COND[c]}\nsonde linéaire sur z (tirets = réseau non entraîné)",
                 fontsize=10)
axs[0][0].set_ylabel("exactitude de la sonde")
axs[0][0].legend(frameon=False, fontsize=8, loc="lower left")
fig.savefig(f"{FIG}/3_sonde_par_n.png", dpi=130, bbox_inches="tight"); plt.close()

# 4. Linear regression z -> n: mean value read out for each quantity, per shape
fig, ax = plt.subplots(figsize=(5.6, 4.6))
ax.plot([1, 9], [1, 9], color=GRIS, lw=1, ls=":", label="lecture parfaite")
for f in FORMES:
    m, s = moy([[r["entraine"]["sondes"][f]["regression_moyenne_par_n"][str(n)] for n in N]
                for r in R[c0]])
    ax.errorbar(N, m, yerr=s, fmt="-o", ms=4, lw=2, color=COUL[f], label=NOM[f], capsize=2)
ax.set_xticks(N); ax.set_yticks(N)
ax.set_xlabel("quantité réelle"); ax.set_ylabel("nombre lu par la droite z → n")
ax.set_title(f"Le nombre est-il une droite dans z ? (condition {c0})", fontsize=10)
ax.legend(frameon=False)
fig.savefig(f"{FIG}/4_regression.png", dpi=130, bbox_inches="tight"); plt.close()

# 5. Trajectory: when does the number appear in z?
fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
for c, rs in R.items():
    rs = [r for r in rs if r.get("trajectoire")]  # eval-only runs do not have one
    if not rs:
        continue
    steps = sorted(map(int, rs[0]["trajectoire"].keys()))
    x = [s + 1 for s in steps]
    for key, ax in [("comparaison", axs[0]), ("sonde", axs[1])]:
        m, s = moy([[r["trajectoire"][str(t)][key] for t in steps] for r in rs])
        ls = {"A": "--", "B": "-", "B2s": ":", "Ase": "-."}[c]
        ax.plot(x, m, ls, marker="o", ms=3, lw=2, color="#2a78d6", label=COND[c])
for ax, t in zip(axs, ["exactitude de comparaison (images de test)",
                       "sonde linéaire « combien ? » sur z"]):
    ax.set_xscale("log"); ax.set_xlabel("pas d'entraînement + 1 (échelle log)"); ax.set_title(t, fontsize=10)
    ax.set_ylim(0, 1.02); ax.legend(frameon=False, fontsize=8)
fig.savefig(f"{FIG}/5_trajectoire.png", dpi=130, bbox_inches="tight"); plt.close()

# 5b. Trajectory per quantity (condition c0, averaged over seeds): small numbers first?
rs = [r for r in R[c0] if r.get("trajectoire")]
steps = sorted(map(int, rs[0]["trajectoire"].keys()))
H = np.mean([[[r["trajectoire"][str(t)]["sonde_par_n"][str(n)] for t in steps] for n in N]
             for r in rs], 0)
fig, ax = plt.subplots(figsize=(7, 3.6))
im = ax.imshow(H, cmap="Blues", vmin=0, vmax=1, aspect="auto", origin="lower")
ax.set_yticks(range(9), N); ax.set_xticks(range(len(steps)), steps, rotation=45)
ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("quantité"); ax.grid(False)
ax.set_title(f"Sonde linéaire par quantité pendant l'entraînement (condition {c0})", fontsize=10)
fig.colorbar(im, ax=ax, shrink=0.9)
fig.savefig(f"{FIG}/5b_trajectoire_par_n.png", dpi=130, bbox_inches="tight"); plt.close()

# 6. 2D PCA of z (seed 0): trained network colored by quantity, then by mode, then the untrained network
d = np.load(f"results/{D}_{c0}_seed0/z.npz")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
panneaux = [(f"z_{D}", "y", "réseau entraîné, couleur = quantité"),
            (f"z_{D}", "mode", "réseau entraîné, couleur = mode (clair : aire fixe, foncé : taille fixe)"),
            (f"init_z_{D}", "y", "réseau NON entraîné, couleur = quantité")]
for ax, (cle, coul, titre) in zip(axs, panneaux):
    pca = PCA(2).fit(d[cle]); p2 = pca.transform(d[cle]); y = d[f"y_{D}"]
    if coul == "y":
        ax.scatter(p2[:, 0], p2[:, 1], c=y, cmap="Blues", vmin=-1, vmax=9, s=4, alpha=0.7, edgecolors="none")
    else:
        ax.scatter(p2[:, 0], p2[:, 1], c=d[f"mode_{D}"], cmap="Blues", vmin=-0.6, vmax=1.2, s=4, alpha=0.7,
                   edgecolors="none")
    for n in N:
        cx, cy = p2[y == n].mean(0)
        ax.text(cx, cy, str(n), fontsize=10, weight="bold", ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="#52514e", lw=0.8))
    ax.set_title(titre, fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_xlabel(f"axe 1 ({100 * pca.explained_variance_ratio_[0]:.0f} % de la variance)")
    ax.set_ylabel(f"axe 2 ({100 * pca.explained_variance_ratio_[1]:.0f} %)")
fig.suptitle("ACP de z (chiffre = centre du nuage de cette quantité)", fontsize=11)
fig.savefig(f"{FIG}/6_acp.png", dpi=130, bbox_inches="tight"); plt.close()

# Numerical summary
for c, rs in R.items():
    print(f"=== condition {c} ({len(rs)} graines)")
    for f in FORMES:
        g = lambda fn: (np.mean([fn(r) for r in rs]), np.std([fn(r) for r in rs]))
        comp = g(lambda r: np.mean([v["acc"] for v in r["entraine"][f]["paires"].values()]))
        ret = g(lambda r: np.mean([v["acc"] for k, v in r["entraine"][f]["paires"].items()
                                   if v["retenue"] and k[0] != k[2]]))
        eg = g(lambda r: np.mean([r["entraine"][f]["paires"][f"{n}-{n}"]["acc"] for n in N]))
        six = g(lambda r: r["entraine"][f]["paires"]["6-6"]["acc"])
        tr = g(lambda r: r["entraine"][f]["transitivite"]["coherence"])
        so = g(lambda r: r["entraine"]["sondes"][f]["classif"])
        so0 = g(lambda r: r["non_entraine"]["sondes"][f]["classif"])
        r2 = g(lambda r: r["entraine"]["sondes"][f]["regression_r2"])
        print(f"[{f:8s}] comparaison {comp[0]:.3f}±{comp[1]:.3f} | inégales jamais vues {ret[0]:.3f} | "
              f"égal (moy.) {eg[0]:.3f} | 6=6 {six[0]:.3f}±{six[1]:.3f} | transitivité {tr[0]:.4f} | "
              f"sonde {so[0]:.3f}±{so[1]:.3f} (non entraîné {so0[0]:.3f}) | R² droite {r2[0]:.3f}")
        if "acc_interforme" in rs[0]["entraine"][f]:
            for cle in ["acc_meme_forme", "acc_interforme"]:
                print(f"   {cle} {np.mean([r['entraine'][f][cle] for r in rs]):.3f}")
            for k in rs[0]["entraine"]["sondes"]:
                if "_" in k:
                    print(f"   sonde {k} {np.mean([r['entraine']['sondes'][k]['classif'] for r in rs]):.3f}")
    print("sonde « quantité de blanc » R² : entraîné",
          round(np.mean([r["entraine"]["sonde_blanc_r2"] for r in rs]), 3),
          "| non entraîné", round(np.mean([r["non_entraine"]["sonde_blanc_r2"] for r in rs]), 3))
