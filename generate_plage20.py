"""Images with 16 to 20 objects (same rules as generate_plage.py) for the addition experiment (sums up to 20).
Usage (from the repository root): uv run python generate_plage20.py
    -> data/train_plage16a20.npz (3000 per number), data/test_plage16a20.npz (500)"""
import numpy as np
from generate_plage import jeu as _jeu
from generate import dessiner

def jeu(n_par_classe, seed):
    rng = np.random.default_rng(seed)
    X, y, m = [], [], []
    for n in range(16, 21):
        for k in range(n_par_classe):
            mode = "aire" if k % 2 == 0 else "taille"
            for _ in range(50):
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
    for nom, npc, seed in [("test_plage16a20", 500, 61), ("train_plage16a20", 3000, 60)]:
        X, y, m = jeu(npc, seed)
        np.savez_compressed(f"data/{nom}.npz", X=X, y=y, mode=m)
        print(nom, X.shape, flush=True)
