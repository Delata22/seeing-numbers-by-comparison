"""Generates data/grand_<version>.npz: 200 images per quantity, from 1 to 30.

Fixed-size mode up to 20 objects for mix, 14 for uni (20 fixed-size triangles do not fit
in 64x64); beyond that, fixed area only.
Usage (from the repository root): uv run python generate_grand.py [mix] [uni]
"""
import sys
import numpy as np
from generate import jeu_grand
CONFIG = {"mix": (30, 20), "uni": (31, 14)}   # seed, max n in fixed-size mode
for forme in (sys.argv[1:] or list(CONFIG)):
    seed, n_taille = CONFIG[forme]
    X, y, m, fs = jeu_grand(200, forme, seed, n_max_taille=n_taille)
    np.savez_compressed(f"data/grand_{forme}.npz", X=X, y=y, mode=m, forme=fs)
    print(f"grand_{forme}", X.shape, flush=True)
