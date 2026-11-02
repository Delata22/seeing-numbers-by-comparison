"""Images for the "training range 1 to N" experiment (N from 3 to 15).
Same rules as generate.jeu (mix version, half fixed area, half fixed size), but with 1 to 15 objects.
Each N uses the subset y <= N of the same images.
Usage (from the repository root): uv run python generate_plage.py
    -> data/train_plage15.npz (6000 per number), data/test_plage15.npz (1000 per number)
"""
import numpy as np
from generate import dessiner

def jeu(n_par_classe, seed, n_max=15):
    rng = np.random.default_rng(seed)
    X, y, m = [], [], []
    for n in range(1, n_max + 1):
        for k in range(n_par_classe):
            mode = "aire" if k % 2 == 0 else "taille"
            for _ in range(20):
                try:
                    X.append(dessiner(n, mode, "mix", rng)); break
                except RuntimeError:
                    pass
            else:
                raise RuntimeError(f"placement impossible n={n}")
            y.append(n); m.append(0 if mode == "aire" else 1)
        print("n =", n, flush=True)
    return np.stack(X), np.array(y), np.array(m)

if __name__ == "__main__":
    for nom, npc, seed in [("test_plage15", 1000, 51), ("train_plage15", 6000, 50)]:
        X, y, m = jeu(npc, seed)
        np.savez_compressed(f"data/{nom}.npz", X=X, y=y, mode=m)
        print(nom, X.shape, flush=True)
