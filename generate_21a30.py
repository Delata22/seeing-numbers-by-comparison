"""Training images with 21 to 30 objects at fixed area (mixed shapes, as in generate.dessiner), for part B
of the "beyond 20" experiment. The test set remains data/grand_mix.npz (different images).
Usage (from the repository root): uv run python generate_21a30.py -> data/train_21a30_aire.npz (2000 per number)"""
import numpy as np
from generate import dessiner
rng = np.random.default_rng(70)
X, y = [], []
for n in range(21, 31):
    for _ in range(2000):
        for _ in range(50):
            try:
                X.append(dessiner(n, "aire", "mix", rng)); break
            except RuntimeError:
                pass
        y.append(n)
    print("n =", n, flush=True)
np.savez_compressed("data/train_21a30_aire.npz", X=np.stack(X), y=np.array(y), mode=np.zeros(len(y), int))
print("fait", len(y))
