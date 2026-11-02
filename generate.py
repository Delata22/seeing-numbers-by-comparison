"""Generates the images: 64x64, black background, 1 to 9 non-overlapping white shapes.

Two modes, so that no "easy" cue gives away the quantity:
- "aire"   (fixed area): fixed total white area -> the more shapes, the smaller they are;
- "taille" (fixed size): fixed size for each shape -> total area grows with the quantity.
In both cases the size gets a small amount of noise (+/-15%).

Shapes: disk ("disque"), square ("carre"), equilateral triangle ("triangle"). At equal
size, all three have the same area (a shape of "radius" r has area pi r^2).
"""
import math
import numpy as np
from PIL import Image, ImageDraw

SIZE = 64
SS = 4                 # supersampling for smooth edges
AIRE_TOTALE = 320.0    # "aire" mode: total white pixels (~8% of the image)
RAYON_FIXE = 3.8       # "taille" mode: radius of each shape (area ~45 px)
ECART = 2.0            # minimum gap between two shapes (pixels)
MARGE = 1.0            # minimum gap to the image border


def _rayons(n, mode, rng):
    if mode == "aire":
        r = math.sqrt(AIRE_TOTALE / n / math.pi)
    else:
        r = RAYON_FIXE
    rs = r * rng.uniform(0.85, 1.15, size=n)
    if mode == "aire":  # renormalize so the total area stays exact
        rs *= math.sqrt(AIRE_TOTALE / (math.pi * (rs ** 2).sum()))
    return rs


def _rayon_englobant(r, forme):
    """Radius of the circle enclosing the shape (used to test overlaps)."""
    if forme == "disque":
        return r
    if forme == "carre":  # side c = r*sqrt(pi), half-diagonal c/sqrt(2)
        return r * math.sqrt(math.pi) / math.sqrt(2)
    # equilateral triangle of area pi r^2: side a = sqrt(4 pi r^2 / sqrt(3)), R = a/sqrt(3)
    a = math.sqrt(4 * math.pi * r * r / math.sqrt(3))
    return a / math.sqrt(3)


def _placer(rs_eng, rng, essais=2000):
    """Rejection sampling: non-overlapping center positions."""
    for _ in range(50):
        centres = []
        ok = True
        for R in rs_eng:
            for _ in range(essais):
                x, y = rng.uniform(R + MARGE, SIZE - R - MARGE, size=2)
                if all(math.hypot(x - cx, y - cy) >= R + cR + ECART for cx, cy, cR in centres):
                    centres.append((x, y, R))
                    break
            else:
                ok = False
                break
        if ok:
            return centres
    raise RuntimeError("placement impossible")


FORMES = ["disque", "carre", "triangle"]


def dessiner(n, mode, forme, rng):
    """forme = "disque", "carre", "triangle", or "mix" (shape drawn at random for each object)."""
    rs = _rayons(n, mode, rng)
    formes = list(rng.choice(FORMES, n)) if forme == "mix" else [forme] * n
    rs_eng = [_rayon_englobant(r, f) for r, f in zip(rs, formes)]
    ordre = np.argsort(-np.array(rs_eng))           # place the largest ones first
    centres = _placer([rs_eng[i] for i in ordre], rng)
    img = Image.new("L", (SIZE * SS, SIZE * SS), 0)
    d = ImageDraw.Draw(img)
    for (x, y, _), i in zip(centres, ordre):
        r, f = rs[i], formes[i]
        X, Y = x * SS, y * SS
        if f == "disque":
            d.ellipse([X - r * SS, Y - r * SS, X + r * SS, Y + r * SS], fill=255)
        else:
            k = 4 if f == "carre" else 3
            R = _rayon_englobant(r, f) * SS
            t0 = rng.uniform(0, 2 * math.pi)
            pts = [(X + R * math.cos(t0 + 2 * math.pi * j / k),
                    Y + R * math.sin(t0 + 2 * math.pi * j / k)) for j in range(k)]
            d.polygon(pts, fill=255)
    img = img.resize((SIZE, SIZE), Image.BOX)
    return np.asarray(img, dtype=np.uint8)


def jeu(n_par_classe, forme, seed):
    """n_par_classe images per quantity 1..9, half in "aire" mode, half in "taille" mode.

    forme = "uni": a single shape per image, the three shapes in equal proportions
    for each (quantity, mode). The image's shape is stored in f (-1 if mixed).
    """
    rng = np.random.default_rng(seed)
    X, y, m, fs = [], [], [], []
    for n in range(1, 10):
        for k in range(n_par_classe):
            mode = "aire" if k % 2 == 0 else "taille"
            f = FORMES[(k // 2) % 3] if forme == "uni" else forme
            X.append(dessiner(n, mode, f, rng))
            y.append(n)
            m.append(0 if mode == "aire" else 1)
            fs.append(FORMES.index(f) if f in FORMES else -1)
    return np.stack(X), np.array(y), np.array(m), np.array(fs)


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    import sys
    if "--ancien" in sys.argv:  # first design: training on disks only
        specs = {"train_disque": (6000, "disque", 0), "test_disque": (1000, "disque", 1),
                 "test_carre": (1000, "carre", 2), "test_triangle": (1000, "triangle", 3)}
    else:  # current design, two versions; same rules for training and test
        specs = {"train_mix": (6000, "mix", 10),   # several shapes in each image
                 "test_mix": (1000, "mix", 11),
                 "train_uni": (6000, "uni", 20),   # one shape per image, all 3 in the dataset
                 "test_uni": (1000, "uni", 21)}
        specs = {k: v for k, v in specs.items() if not sys.argv[1:] or k.split("_")[1] in sys.argv[1:]}
    for nom, (npc, forme, seed) in specs.items():
        X, y, m, fs = jeu(npc, forme, seed)
        np.savez_compressed(f"data/{nom}.npz", X=X, y=y, mode=m, forme=fs)
        print(nom, X.shape)


def jeu_grand(n_par_classe, forme, seed, n_max=30, n_max_taille=20):
    """Test set for large numbers: 1 to n_max objects.
    Up to n_max_taille: half fixed area, half fixed size (as in training).
    Beyond that: fixed area only (at fixed size, the shapes no longer fit in 64x64)."""
    rng = np.random.default_rng(seed)
    X, y, m, fs = [], [], [], []
    for n in range(1, n_max + 1):
        for k in range(n_par_classe):
            mode = "aire" if (k % 2 == 0 or n > n_max_taille) else "taille"
            f = FORMES[(k // 2) % 3] if forme == "uni" else forme
            for essai in range(20):  # placement sometimes fails with 20 fixed-size shapes: retry
                try:
                    X.append(dessiner(n, mode, f, rng)); break
                except RuntimeError:
                    pass
            else:
                raise RuntimeError(f"placement impossible : n={n}, mode={mode}")
            y.append(n); m.append(0 if mode == "aire" else 1)
            fs.append(FORMES.index(f) if f in FORMES else -1)
    return np.stack(X), np.array(y), np.array(m), np.array(fs)
