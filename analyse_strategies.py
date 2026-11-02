"""Why does generalization improve with N, and do the models share the same strategy?

"Range 1 to N" models (x1/8, z = 1, head 128), N = 2, 3, 5, 7, 9, 12, 15, with and without "equal", 3 seeds.
Read-only, on CPU. Three analyses:
  1. Weight drift from initialization, layer by layer (same seed = same init).
  2. Layer by layer: is the number linearly present up to 20 (probe trained on 1..20),
     and does the code stay linear beyond N (probe trained on 1..N only, read at 20)?
  3. Representation similarity (linear CKA) between models, layer by layer.

Outputs: logs/strategies/resume.txt, figures/strategies/*.png
Usage (from the repository root): uv run python analyse_strategies.py
"""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import math
from PIL import Image, ImageDraw
import generate as G
from reduction import Comparateur


def scene(n, rng):
    """Like generate.dessiner(n, 'taille', 'mix') (copy of analyse_poids.scene)."""
    rs = G._rayons(n, "taille", rng)
    formes = list(rng.choice(G.FORMES, n))
    rs_eng = [G._rayon_englobant(r, f) for r, f in zip(rs, formes)]
    ordre = np.argsort(-np.array(rs_eng))
    try:
        centres = G._placer([rs_eng[i] for i in ordre], rng)
    except RuntimeError:
        return scene(n, rng)
    SS, SIZE = G.SS, G.SIZE
    img = Image.new("L", (SIZE * SS, SIZE * SS), 0)
    d = ImageDraw.Draw(img)
    for (x, y, _), i in zip(centres, ordre):
        r, f = rs[i], formes[i]
        X_, Y_ = x * SS, y * SS
        if f == "disque":
            d.ellipse([X_ - r * SS, Y_ - r * SS, X_ + r * SS, Y_ + r * SS], fill=255)
        else:
            k = 4 if f == "carre" else 3
            R = G._rayon_englobant(r, f) * SS
            t0 = rng.uniform(0, 2 * math.pi)
            d.polygon([(X_ + R * math.cos(t0 + 2 * math.pi * j / k), Y_ + R * math.sin(t0 + 2 * math.pi * j / k))
                       for j in range(k)], fill=255)
    return np.asarray(img.resize((SIZE, SIZE), Image.BOX), dtype=np.uint8), None

torch.set_num_threads(8)
NS = [2, 3, 5, 7, 9, 12, 15]
os.makedirs("logs/strategies", exist_ok=True)
os.makedirs("figures/strategies", exist_ok=True)
out = open("logs/strategies/resume.txt", "w")


def ecrire(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.write(s + "\n")


def nom(N, egal):
    return f"plage_N{N}_x1-8_z1_tete128_{'egal' if egal else 'sans_egal'}"


def charger(N, egal, seed):
    m = Comparateur(8, 1, 3 if egal else 2, 128)
    m.load_state_dict(torch.load(f"modeles/{nom(N, egal)}/seed{seed}.pt", map_location="cpu"))
    return m.eval()


def init(egal, seed):
    torch.manual_seed(seed)
    return Comparateur(8, 1, 3 if egal else 2, 128).eval()


rng = np.random.default_rng(321)
PAR_N = 80
imgs, ys = [], []
for n in range(1, 21):
    for _ in range(PAR_N):
        im, _ = scene(n, rng)
        imgs.append(im); ys.append(n)
X = torch.tensor(np.stack(imgs), dtype=torch.float32).unsqueeze(1) / 255.0
Y = np.array(ys)
moitie = rng.permutation(len(Y)) % 2 == 0  # A = probe training, B = test


@torch.no_grad()
def activations(m):
    """Output of each conv+BN+ReLU block (6 layers), flattened; plus z."""
    acts, h = [], X
    for i in range(0, len(m.enc.conv), 3):
        h = m.enc.conv[i + 2](m.enc.conv[i + 1](m.enc.conv[i](h)))
        acts.append(h.flatten(1).numpy().astype(np.float64))
    acts.append(m.enc.proj(h.flatten(1)).numpy().astype(np.float64))
    return acts


def ridge(A_tr, y_tr, A_te, alpha_rel=1e-3):
    """Ridge regression in dual form (few examples, many dimensions)."""
    mu = A_tr.mean(0)
    Ac, Tc = A_tr - mu, A_te - mu
    K = Ac @ Ac.T
    a = alpha_rel * np.trace(K) / len(K) + 1e-9
    w = Ac.T @ np.linalg.solve(K + a * np.eye(len(K)), y_tr - y_tr.mean())
    return Tc @ w + y_tr.mean()


def cka(A, B):
    A = A - A.mean(0); B = B - B.mean(0)
    Ka, Kb = A @ A.T, B @ B.T
    return float((Ka * Kb).sum() / np.sqrt((Ka * Ka).sum() * (Kb * Kb).sum()))


COUCHES = ["c1", "c2", "c3", "c4", "c5", "c6", "z"]
res = {}
for egal in (True, False):
    for seed in range(3):
        m0 = init(egal, seed)
        p0 = [p.detach().clone() for p in m0.enc.parameters()]
        for N in NS:
            m = charger(N, egal, seed)
            r = {}
            # 1. relative weight drift, per layer (conv and projection weights)
            mods = [m.enc.conv[i] for i in range(0, len(m.enc.conv), 3)] + [m.enc.proj]
            mods0 = [m0.enc.conv[i] for i in range(0, len(m0.enc.conv), 3)] + [m0.enc.proj]
            r["derive"] = [float((a.weight - b.weight).norm() / b.weight.norm()) for a, b in zip(mods, mods0)]
            # 2. linear probes, layer by layer
            acts = activations(m)
            info, extra = [], []
            for A in acts:
                pred = ridge(A[moitie], Y[moitie].astype(float), A[~moitie])
                ss = ((pred - Y[~moitie]) ** 2).sum() / ((Y[~moitie] - Y[~moitie].mean()) ** 2).sum()
                info.append(1 - ss)
                sel = moitie & (Y <= N)
                pred2 = ridge(A[sel], Y[sel].astype(float), A[~moitie & (Y == 20)])
                extra.append(float(pred2.mean()))
            r["info_r2"], r["lit20_sonde_1N"] = info, extra
            r["acts"] = acts
            res[(egal, seed, N)] = r
        res[(egal, seed, "init")] = {"acts": activations(m0)}
        ecrire(f"fini : {'avec' if egal else 'sans'} égal, graine {seed}")

ecrire("\n=== 1. Dérive des poids depuis l'init, ‖W − W0‖ / ‖W0‖ (moyenne sur 3 graines) ===")
for egal in (True, False):
    ecrire(f"--- {'avec' if egal else 'sans'} égal --- couches : " + " ".join(f"{c:>5s}" for c in COUCHES))
    for N in NS:
        d = np.mean([res[(egal, s, N)]["derive"] for s in range(3)], 0)
        ecrire(f"N={N:2d}  " + " ".join(f"{v:5.2f}" for v in d))

ecrire("\n=== 2a. Le nombre est-il présent linéairement jusqu'à 20 ? R² d'une sonde apprise sur 1..20 ===")
for egal in (True, False):
    ecrire(f"--- {'avec' if egal else 'sans'} égal --- " + " ".join(f"{c:>5s}" for c in COUCHES))
    for N in NS:
        d = np.mean([res[(egal, s, N)]["info_r2"] for s in range(3)], 0)
        ecrire(f"N={N:2d}  " + " ".join(f"{v:5.3f}" for v in d))

ecrire("\n=== 2b. Le code reste-t-il linéaire au-delà de N ? Sonde apprise sur 1..N, lecture moyenne de 20 ===")
for egal in (True, False):
    ecrire(f"--- {'avec' if egal else 'sans'} égal --- " + " ".join(f"{c:>5s}" for c in COUCHES))
    for N in NS:
        d = np.mean([res[(egal, s, N)]["lit20_sonde_1N"] for s in range(3)], 0)
        ecrire(f"N={N:2d}  " + " ".join(f"{v:5.1f}" for v in d))

# 3. CKA
ecrire("\n=== 3. Similarité CKA, couche par couche (moyenne sur graines / paires de graines) ===")
paires = {
    "N3 vs N15, même graine": lambda e: [(res[(e, s, 3)], res[(e, s, 15)]) for s in range(3)],
    "N15, graines différentes": lambda e: [(res[(e, a, 15)], res[(e, b, 15)]) for a, b in ((0, 1), (0, 2), (1, 2))],
    "N3, graines différentes": lambda e: [(res[(e, a, 3)], res[(e, b, 3)]) for a, b in ((0, 1), (0, 2), (1, 2))],
    "init vs N15, même graine": lambda e: [(res[(e, s, "init")], res[(e, s, 15)]) for s in range(3)],
    "init, graines différentes": lambda e: [(res[(e, a, "init")], res[(e, b, "init")]) for a, b in ((0, 1), (0, 2), (1, 2))],
    "avec vs sans égal, N15, même graine": None,
}
sub = np.arange(len(Y)) % 2 == 0  # 800 images for CKA
for egal in (True, False):
    ecrire(f"--- {'avec' if egal else 'sans'} égal --- " + " ".join(f"{c:>5s}" for c in COUCHES))
    for lab, f in paires.items():
        if f is None:
            continue
        v = np.mean([[cka(a["acts"][l][sub], b["acts"][l][sub]) for l in range(7)] for a, b in f(egal)], 0)
        ecrire(f"{lab:38s} " + " ".join(f"{x:5.2f}" for x in v))
v = np.mean([[cka(res[(True, s, 15)]["acts"][l][sub], res[(False, s, 15)]["acts"][l][sub]) for l in range(7)] for s in range(3)], 0)
ecrire(f"{'avec vs sans égal, N15, même graine':38s} " + " ".join(f"{x:5.2f}" for x in v))

# CKA matrix across N, layers c6 and z, seed 0
for l, lab in ((5, "c6"), (6, "z")):
    for egal in (True, False):
        M = np.array([[cka(res[(egal, 0, a)]["acts"][l][sub], res[(egal, 0, b)]["acts"][l][sub]) for b in NS] for a in NS])
        ecrire(f"\nCKA entre N, couche {lab}, {'avec' if egal else 'sans'} égal, graine 0 :")
        ecrire("       " + " ".join(f"N{n:<4d}" for n in NS))
        for n, row in zip(NS, M):
            ecrire(f"N{n:<5d} " + " ".join(f"{x:5.2f}" for x in row))
out.close()

# figure: readout of 20 by the 1..N probe, layer by layer
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for k, egal in enumerate((True, False)):
    for N in NS:
        d = np.mean([res[(egal, s, N)]["lit20_sonde_1N"] for s in range(3)], 0)
        ax[k].plot(COUCHES, d, marker="o", label=f"N={N}")
    ax[k].axhline(20, color="k", lw=0.8, ls="--")
    ax[k].set_title(f"{'Avec' if egal else 'Sans'} égal : sonde apprise sur 1..N, lecture de 20")
    ax[k].set_xlabel("couche"); ax[k].legend(fontsize=8)
ax[0].set_ylabel("20 lu comme")
plt.tight_layout()
plt.savefig("figures/strategies/linearite_par_couche.png", dpi=110)
print("fini")
