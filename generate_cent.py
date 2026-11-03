"""Pre-generated dataset for the 1-100 series (fallback: on-the-fly generation costs 0.37 s
per 256 images, which is too slow). 300 images per number from 1 to 100, generated on the GPU
with cent.generer. Labels are the numbers of objects actually placed.
Usage (from the repository root): uv run python generate_cent.py
    -> data/cent_X.npy (uint8, 30000x192x192), data/cent_y.npy
"""
import numpy as np, torch
from cent import generer, N_MAX
dev = "cuda"; g = torch.Generator(device=dev); g.manual_seed(31337)
ns = torch.arange(1, N_MAX + 1, device=dev).repeat_interleave(300)
ns = ns[torch.randperm(len(ns), device=dev, generator=g)]
Xs, ys = [], []
for i in range(0, len(ns), 512):
    X, y = generer(ns[i:i + 512], g, dev); Xs.append(X[:, 0].cpu()); ys.append(y.cpu())
X = torch.cat(Xs).numpy(); y = torch.cat(ys).numpy()
np.save("data/cent_X.npy", X); np.save("data/cent_y.npy", y)
print(X.shape, "échecs de placement :", int((y != ns.cpu().numpy()).sum()))
