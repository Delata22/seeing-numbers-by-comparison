"""Analysis of the "range 1 to N" models (x1/8, z = 1): filters and mechanism.

Read-only use of the models saved in modeles/. Everything runs on CPU.
Common unit: the "readout" = z converted into a number of objects by the line
z -> n fitted on 1..N (fixed-size images), specific to each model.

Outputs: figures/poids/*.png and logs/poids/resume.txt
Usage (from the repository root): uv run python analyse_poids.py
"""
import math
import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import generate as G
from reduction import Comparateur

torch.set_num_threads(8)
NS = [3, 5, 7, 9, 12, 15]
SS, SIZE = G.SS, G.SIZE
os.makedirs("figures/poids", exist_ok=True)
os.makedirs("logs/poids", exist_ok=True)
sortie = open("logs/poids/resume.txt", "w")


def ecrire(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    sortie.write(s + "\n")


def charger(N, egal, seed):
    nom = f"plage_N{N}_x1-8_z1_tete128_{'egal' if egal else 'sans_egal'}"
    m = Comparateur(8, 1, 3 if egal else 2, 128)
    m.load_state_dict(torch.load(f"modeles/{nom}/seed{seed}.pt", map_location="cpu"))
    return m.eval()


def rendre(objets):
    """objets = [(x, y, r)] discs; same rendering as generate.dessiner."""
    img = Image.new("L", (SIZE * SS, SIZE * SS), 0)
    d = ImageDraw.Draw(img)
    for x, y, r in objets:
        X, Y = x * SS, y * SS
        d.ellipse([X - r * SS, Y - r * SS, X + r * SS, Y + r * SS], fill=255)
    return np.asarray(img.resize((SIZE, SIZE), Image.BOX), dtype=np.uint8)


def scene(n, rng):
    """Like dessiner(n, 'taille', 'mix') but also returns the centres."""
    rs = G._rayons(n, "taille", rng)
    formes = list(rng.choice(G.FORMES, n))
    rs_eng = [G._rayon_englobant(r, f) for r, f in zip(rs, formes)]
    ordre = np.argsort(-np.array(rs_eng))
    try:
        centres = G._placer([rs_eng[i] for i in ordre], rng)
    except RuntimeError:  # 20 shapes: placement sometimes impossible, redraw
        return scene(n, rng)
    img = Image.new("L", (SIZE * SS, SIZE * SS), 0)
    d = ImageDraw.Draw(img)
    for (x, y, _), i in zip(centres, ordre):
        r, f = rs[i], formes[i]
        X, Y = x * SS, y * SS
        if f == "disque":
            d.ellipse([X - r * SS, Y - r * SS, X + r * SS, Y + r * SS], fill=255)
        else:
            k = 4 if f == "carre" else 3
            R = G._rayon_englobant(r, f) * SS
            t0 = rng.uniform(0, 2 * math.pi)
            d.polygon([(X + R * math.cos(t0 + 2 * math.pi * j / k), Y + R * math.sin(t0 + 2 * math.pi * j / k))
                       for j in range(k)], fill=255)
    img = np.asarray(img.resize((SIZE, SIZE), Image.BOX), dtype=np.uint8)
    return img, [(x, y) for x, y, _ in centres]


@torch.no_grad()
def z_de(m, imgs):
    x = torch.tensor(np.stack(imgs), dtype=torch.float32).unsqueeze(1) / 255.0
    return m.enc(x).squeeze(1).numpy()


@torch.no_grad()
def carte_finale(m, imgs):
    x = torch.tensor(np.stack(imgs), dtype=torch.float32).unsqueeze(1) / 255.0
    return m.enc.conv(x).numpy()  # (B, C, 4, 4)


def filtres_effectifs(m):
    """First convolution with the BatchNorm folded into it."""
    conv, bn = m.enc.conv[0], m.enc.conv[1]
    g = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
    W = (conv.weight.detach()[:, 0] * g[:, None, None]).numpy()
    b = ((conv.bias.detach() - bn.running_mean) * g + bn.bias.detach()).numpy()
    return W, b


# --- reference images (same for every model) ---------------------------------
rng = np.random.default_rng(123)
REF = {n: [scene(n, rng) for _ in range(150)] for n in range(1, 21)}
CELL_IMGS = [scene(int(rng.integers(1, 10)), rng) for _ in range(1500)]

modeles = {(N, e, s): charger(N, e, s) for N in NS for e in (True, False) for s in range(3)}
resultats = {}

for (N, egal, seed), m in modeles.items():
    r = {}
    zs = {n: z_de(m, [im for im, _ in REF[n]]) for n in range(1, 21)}
    moy = np.array([zs[n].mean() for n in range(1, 21)])
    a, b = np.polyfit(np.arange(1, N + 1), moy[:N], 1)
    lire = lambda z: (np.asarray(z) - b) / a
    z0 = z_de(m, [rendre([])])[0]
    r["vide"] = float(lire(z0))
    r["lecture_moy"] = lire(moy)
    r["lecture_ecart"] = np.array([lire(zs[n]).std() for n in range(1, 21)])

    # T1: a single disc, moved across the whole image
    pos = [(x, y) for x in np.arange(5, 60, 3) for y in np.arange(5, 60, 3)]
    l1 = lire(z_de(m, [rendre([(x, y, 3.8)]) for x, y in pos]))
    r["un_objet"] = (l1.mean(), l1.std(), l1.min(), l1.max())
    r["carte_un_objet"] = l1.reshape(len(np.arange(5, 60, 3)), -1)

    # T2: one disc of increasing size (random positions)
    tailles = [1.0, 1.5, 2.0, 2.5, 3.0, 3.8, 5.0, 6.5, 8.0, 10.0]
    rr = np.random.default_rng(7)
    r["taille"] = [float(lire(z_de(m, [rendre([(*rr.uniform(12, 52, 2), t)]) for _ in range(60)])).mean())
                   for t in tailles]

    # T3: two discs moved closer together (distance between centres)
    dists = [3.0, 5.0, 6.5, 7.6, 8.5, 10.0, 12.0, 15.0, 20.0]
    vals = []
    for dd in dists:
        imgs = []
        for _ in range(60):
            cx, cy = rr.uniform(18, 46, 2)
            t = rr.uniform(0, 2 * math.pi)
            dx, dy = dd / 2 * math.cos(t), dd / 2 * math.sin(t)
            imgs.append(rendre([(cx - dx, cy - dy, 3.8), (cx + dx, cy + dy, 3.8)]))
        vals.append(float(lire(z_de(m, imgs)).mean()))
    r["fusion"] = vals

    # T4: does the last layer compute a sum over cells?
    W = m.enc.proj.weight.detach().numpy().reshape(-1, 4, 4)       # (C, 4, 4)
    A = carte_finale(m, [im for im, _ in CELL_IMGS])               # (B, C, 4, 4)
    contrib = (A * W[None]).sum(1)                                 # (B, 4, 4) in units of z
    comptes = np.zeros((len(CELL_IMGS), 4, 4))
    for i, (_, cs) in enumerate(CELL_IMGS):
        for x, y in cs:
            comptes[i, min(int(y // 16), 3), min(int(x // 16), 3)] += 1
    c, k = contrib.reshape(-1) / a, comptes.reshape(-1)
    pente, ordo = np.polyfit(k, c, 1)
    r2 = 1 - ((c - (pente * k + ordo)) ** 2).sum() / ((c - c.mean()) ** 2).sum()
    pentes_case = [np.polyfit(comptes[:, i, j], contrib[:, i, j] / a, 1)[0] for i in range(4) for j in range(4)]
    r["cases"] = (r2, pente, np.std(pentes_case) / abs(np.mean(pentes_case)))
    r["filtres"] = filtres_effectifs(m)
    resultats[(N, egal, seed)] = r

# --- summary ------------------------------------------------------------------
ecrire("Analyse des modèles plage 1-N (×1/8, z = 1, tête 128). Moyenne [min-max] sur 3 graines.")
ecrire("Lecture = z converti en objets par la droite ajustée sur 1..N (taille fixe).\n")


def mm(v):
    v = np.array(v)
    return f"{v.mean():.2f} [{v.min():.2f}-{v.max():.2f}]"


for egal in (True, False):
    ecrire(f"=== {'AVEC' if egal else 'SANS'} « égal » ===")
    for N in NS:
        R = [resultats[(N, egal, s)] for s in range(3)]
        ecrire(f"N={N:2d} | vide lu {mm([x['vide'] for x in R])} | un objet n'importe où : "
               f"{mm([x['un_objet'][0] for x in R])} (écart selon la position {mm([x['un_objet'][1] for x in R])})")
        ecrire(f"      taille (r=1..10) : " + " ".join(f"{np.mean([x['taille'][i] for x in R]):.2f}" for i in range(10)))
        ecrire(f"      2 disques, distance 3..20 : " + " ".join(f"{np.mean([x['fusion'][i] for x in R]):.2f}" for i in range(9)))
        ecrire(f"      cases 4x4 : R² contribution~objets {mm([x['cases'][0] for x in R])}, pente {mm([x['cases'][1] for x in R])}, "
               f"variation des pentes entre cases {mm([x['cases'][2] for x in R])}")
        lm = np.mean([x["lecture_moy"] for x in R], 0)
        le = np.mean([x["lecture_ecart"] for x in R], 0)
        ecrire(f"      lecture n=1..20 : " + " ".join(f"{v:.1f}" for v in lm))
        ecrire(f"      étalement n=1..20 : " + " ".join(f"{v:.2f}" for v in le))
        n = np.arange(1, 21)
        alpha = np.polyfit(np.log(n[:N + 3]), np.log(le[:N + 3] + 1e-9), 1)[0]
        ecrire(f"      étalement ~ n^{alpha:.2f} sur 1..{N + 3}  (0.5 = somme de détections bruitées, 1 = Weber)")
    ecrire("")

# similarity of first-layer filters across models (best matching)
ecrire("=== Filtres de la 1re couche (BN repliée) : corrélation moyenne après appariement ===")
cles = [(N, e, 0) for e in (True, False) for N in NS]


def sim(k1, k2):
    W1, W2 = resultats[k1]["filtres"][0].reshape(4, -1), resultats[k2]["filtres"][0].reshape(4, -1)
    C = np.corrcoef(np.vstack([W1, W2]))[:4, 4:]
    i, j = linear_sum_assignment(-C)
    return C[i, j].mean()


S = np.array([[sim(a, b) for b in cles] for a in cles])
noms = [f"{'eg' if e else 'se'}{N}" for N, e, _ in cles]
ecrire("      " + " ".join(f"{x:>5s}" for x in noms))
for nm, row in zip(noms, S):
    ecrire(f"{nm:>5s} " + " ".join(f"{v:5.2f}" for v in row))
for k in cles:
    W, b = resultats[k]["filtres"]
    desc = []
    for f, bb in zip(W, b):
        centre, bord = f[1, 1], (f.sum() - f[1, 1]) / 8
        desc.append(f"somme {f.sum():+.2f} centre {centre:+.2f} pourtour {bord:+.2f} biais {bb:+.2f}")
    ecrire(f"{'eg' if k[1] else 'se'}{k[0]} : " + " | ".join(desc))
sortie.close()

# --- figures ------------------------------------------------------------------
fig, ax = plt.subplots(len(cles), 4, figsize=(6, 1.5 * len(cles)))
for r_, k in enumerate(cles):
    W, b = resultats[k]["filtres"]
    v = np.abs(W).max()
    for c_ in range(4):
        ax[r_, c_].imshow(W[c_], cmap="RdBu_r", vmin=-v, vmax=v)
        ax[r_, c_].set_xticks([]); ax[r_, c_].set_yticks([])
        ax[r_, c_].set_title(f"b={b[c_]:+.1f}", fontsize=7)
    ax[r_, 0].set_ylabel(f"{'avec' if k[1] else 'sans'} égal\nN={k[0]}", fontsize=7)
plt.suptitle("Filtres 3x3 de la 1re couche (graine 0, BN repliée)\nrouge = poids positif, bleu = négatif", fontsize=9)
plt.tight_layout()
plt.savefig("figures/poids/filtres_couche1.png", dpi=130)

fig, ax = plt.subplots(2, 3, figsize=(12, 7))
for col, egal in enumerate((True, False)):
    for N in NS:
        R = [resultats[(N, egal, s)] for s in range(3)]
        ax[col, 0].plot(range(1, 21), np.mean([x["lecture_moy"] for x in R], 0), label=f"N={N}")
        ax[col, 1].plot([1.0, 1.5, 2.0, 2.5, 3.0, 3.8, 5.0, 6.5, 8.0, 10.0], np.mean([x["taille"] for x in R], 0), label=f"N={N}")
        ax[col, 2].plot([3.0, 5.0, 6.5, 7.6, 8.5, 10.0, 12.0, 15.0, 20.0], np.mean([x["fusion"] for x in R], 0), label=f"N={N}")
    ax[col, 0].plot([1, 20], [1, 20], "k--", lw=0.8)
    ax[col, 0].set_title(f"{'Avec' if egal else 'Sans'} égal : lecture de n (taille fixe)")
    ax[col, 0].set_xlabel("vrai n"); ax[col, 0].set_ylabel("lu")
    ax[col, 1].set_title("Un seul disque de rayon croissant"); ax[col, 1].set_xlabel("rayon (px)")
    ax[col, 1].axvline(3.8, color="gray", lw=0.6)
    ax[col, 2].set_title("Deux disques (r = 3.8) qu'on rapproche"); ax[col, 2].set_xlabel("distance entre centres (px)")
    ax[col, 2].axvline(7.6, color="gray", lw=0.6)
    for a_ in ax[col]:
        a_.legend(fontsize=7)
plt.tight_layout()
plt.savefig("figures/poids/mecanisme.png", dpi=110)

fig, ax = plt.subplots(2, 6, figsize=(15, 5))
for row, egal in enumerate((True, False)):
    for col, N in enumerate(NS):
        im = ax[row, col].imshow(resultats[(N, egal, 0)]["carte_un_objet"].T, origin="upper", cmap="viridis", vmin=0, vmax=2)
        ax[row, col].set_title(f"{'avec' if egal else 'sans'} égal, N={N}", fontsize=8)
        ax[row, col].set_xticks([]); ax[row, col].set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.6, label="objets lus pour UN disque")
plt.suptitle("Un seul disque placé à chaque position : combien d'objets le modèle lit-il ?")
plt.savefig("figures/poids/un_objet_position.png", dpi=110)
print("fini")
