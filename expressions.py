"""Expressions with images: each side is a sequence of terms (images) joined by learned symbols.

Examples: [3]+[5] ? [8] -> = ; [2]+[4]+[1] ? [3]+[3] -> > ; [5]+[1]+[1] ? [7] -> =.
Expected answer: comparison of the object totals (left > / = / < right). No number labels.

Architecture
- Encoder and MLP comparison head from the starting model (range series, head 128, "equal"), which keep learning
  (encoder at rate 2e-4, head at rate 2e-3).
- Learned symbols (dimension 4): e_plus (and e_moins, "minus", for series 2).
- Recurrent brick h, the same at every step: ReLU MLP [s, z_k, e_op] (1+1+4) -> 128 -> 1.
  Read left to right: s = z_1, then s <- h(s, z_k, e_op_k). The value of a side is the final s.
- Comparison head on (left value, right value): a single number goes straight through (s = z_1).

Training: 1 to 3 terms per side; types drawn in equal shares: number/number, sum/number, number/sum, sum/sum.
1/3 equal totals; otherwise minimum gap 30%, 20%, 10%, then +/-1 (accuracy over the last 300 batches > 90%);
then the term range grows by one number up to 20; 500 consolidation steps (cap 9000).
Variants: "general" (arbitrary terms) or "succ" (arbitrary first term, then only + [1]).

Usage (from the repository root): uv run python expressions.py --depart 3 --variante general --seed 0
Writes results/expressions/D<N>_<variante>_seed<s>/{metrics.json, modele.pt, h.pt}
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from run import augmenter
from compare import GAUCHE, EGAL, DROITE, cible, flottant
from reduction import Comparateur, encoder
from plage import mesures_au_dela
from addition import charger_pool, tirer

T_MAX = 20                       # largest term (images up to 20 objects)
L_MAX = 3                        # terms per side during training
NIVEAUX = [0.3, 0.2, 0.1, 0.0]
PLUS, MOINS = 0, 1


class Brique(nn.Module):
    """h(s, z, e_op) and the symbol vectors."""
    def __init__(self, n_ops=1, d_op=4, cache=128):
        super().__init__()
        self.e = nn.Parameter(torch.randn(n_ops, d_op) * 0.5)
        self.f = nn.Sequential(nn.Linear(2 + d_op, cache), nn.ReLU(), nn.Linear(cache, 1))

    def pas(self, s, z, op):
        return self.f(torch.cat([s, z, self.e[op]], 1))

    def valeur(self, Z, OPS, L):
        """Z: (B, Lmax) term latents; OPS: (B, Lmax) symbol before each term (ignored for the 1st); L: (B,)."""
        s = Z[:, :1]
        for k in range(1, Z.shape[1]):
            sk = self.pas(s, Z[:, k:k + 1], OPS[:, k])
            s = torch.where((L > k)[:, None], sk, s)
        return s


def decomposer(total, L, A, rng, variante):
    """Terms (list) summing to 'total', with L terms in 0..A; None if impossible."""
    if variante == "succ":
        t0 = total - (L - 1)
        return [t0] + [1] * (L - 1) if 0 <= t0 <= A else None
    if total > L * A:
        return None
    for _ in range(20):
        cuts = np.sort(rng.integers(0, total + 1, L - 1))
        t = np.diff(np.concatenate([[0], cuts, [total]]))
        if (t <= A).all():
            return list(t)
    return None


def tirer_exemples(nb, A, niveau, variante, rng):
    """Returns lists of left and right terms (and their totals)."""
    G, D = [], []
    while len(G) < nb:
        typ = rng.integers(4)                      # 0 n/n, 1 sum/n, 2 n/sum, 3 sum/sum
        Lg = 1 if typ in (0, 2) else rng.integers(2, L_MAX + 1)
        Ld = 1 if typ in (0, 1) else rng.integers(2, L_MAX + 1)
        cap_g = A if Lg == 1 else Lg * A
        if variante == "succ":
            cap_g = A if Lg == 1 else A + Lg - 1
        Tg = int(rng.integers(0, cap_g + 1))
        g = decomposer(Tg, Lg, A, rng, variante)
        if g is None:
            continue
        egal = rng.random() < 1 / 3 or (typ == 3 and rng.random() < 0.3)
        if egal:
            Td = Tg
        else:
            dmin = max(1, int(round(NIVEAUX[niveau] * max(Tg, 1))))
            if typ == 3 and rng.random() < 0.5:
                dmin = 1
            k = dmin + int(rng.integers(0, max(1, round(0.3 * max(Tg, 1))) + 1))
            Td = Tg + (k if rng.random() < 0.5 else -k)
        d = decomposer(Td, Ld, A, rng, variante) if Td >= 0 else None
        if d is None:
            continue
        G.append(g); D.append(d)
    return G, D


def tableau(listes, Lmax=None):
    Lmax = Lmax or max(len(x) for x in listes)
    T = np.zeros((len(listes), Lmax), int); L = np.array([len(x) for x in listes])
    for i, x in enumerate(listes):
        T[i, :len(x)] = x
    return T, L


def valeurs_depuis_z(h, zpool, it, listes, rng, dev, ops=None):
    """Value of each side (list of terms) from the precomputed latents of the test images."""
    T, L = tableau(listes)
    idx = np.vectorize(lambda n: rng.choice(it[n]))(T)
    Z = zpool[torch.tensor(idx, device=dev)]                    # (B, Lmax)
    OPS = torch.zeros_like(Z, dtype=torch.long) if ops is None else torch.tensor(ops, device=dev)
    return h.valeur(Z, OPS, torch.tensor(L, device=dev))


@torch.no_grad()
def mesurer(m, h, N, T, G, rng, dev):
    m.eval()
    Xt, yt, it = T
    zt = encoder(m, Xt)[:, 0]
    res = {"ligne": mesures_au_dela(m, N, *G, rng, n_min=0)}
    moy = np.array([zt[torch.tensor(it[n], device=dev)].mean().item() for n in range(T_MAX + 1)])
    k, b = np.polyfit(np.arange(T_MAX + 1), moy, 1)
    lire = lambda v: (v - b) / k
    res["z_par_n"] = {"moyenne": moy.tolist(), "k": float(k), "b": float(b)}
    img = lambda n, nb: zt[torch.tensor(rng.choice(it[n], nb), device=dev)][:, None]

    def juger(Gl, Dl):
        vg = valeurs_depuis_z(h, zt, it, Gl, rng, dev); vd = valeurs_depuis_z(h, zt, it, Dl, rng, dev)
        l = m.comparer(vg, vd).float()
        tg = np.array([sum(x) for x in Gl]); td = np.array([sum(x) for x in Dl])
        p = l.argmax(1).cpu().numpy()
        cote = ((l[:, GAUCHE] > l[:, DROITE]).cpu().numpy() == (tg > td))
        return p, cote, tg, td

    def bloc(Gl, Dl):
        p, cote, tg, td = juger(Gl, Dl)
        d = np.abs(tg - td)
        r = {}
        for nom, sel in [("egal", d == 0), ("ecart1", d == 1), ("ecart2-3", (d >= 2) & (d <= 3)), ("ecart>=5", d >= 5)]:
            if sel.any():
                r[nom] = {"exact": float((p[sel] == cible(tg[sel], td[sel])).mean())}
                if nom != "egal":
                    r[nom]["cote"] = float(cote[sel].mean())
        return r

    def paires(nb, Lg, Ld, tmax, total_min=0, total_max=10 ** 6, succ=False):
        Gl, Dl = [], []
        while len(Gl) < nb:
            g = [int(rng.integers(0, tmax + 1))] + ([1] * (Lg - 1) if succ else list(rng.integers(0, tmax + 1, Lg - 1)))
            Tg = sum(g)
            if not (total_min <= Tg <= total_max):
                continue
            dlt = int(rng.choice([0, 0, -1, 1, -2, 2, -3, 3, -5, 5]))
            Td = Tg + dlt
            dd = decomposer(Td, Ld, T_MAX, rng, "general") if Td >= 0 else None
            if dd is None:
                continue
            Gl.append(g); Dl.append(dd)
        return Gl, Dl

    # 1. scale
    v = h.valeur(torch.cat([img(10, 500), img(10, 500)], 1), torch.zeros(500, 2, dtype=torch.long, device=dev),
                 torch.full((500,), 2, device=dev))
    p20 = m.comparer(v, img(20, 500)).argmax(1)
    p20r = m.comparer(img(20, 500), v).argmax(1)
    a = rng.integers(0, T_MAX + 1, 1000)
    za = zt[torch.tensor([rng.choice(it[x]) for x in a], device=dev)][:, None]
    ha0 = h.valeur(torch.cat([za, img(0, 1000)], 1), torch.zeros(1000, 2, dtype=torch.long, device=dev), torch.full((1000,), 2, device=dev))
    res["echelle"] = {"10+10_lu": float(lire(v.mean().item())), "10+10_vs_20_egal_gauche": float((p20 == EGAL).float().mean()),
                      "10+10_vs_20_egal_droite": float((p20r == EGAL).float().mean()),
                      "a+0_moins_a_en_pas": float(((ha0 - za).abs().mean() / abs(k)).item())}
    # 2. forgetting: image vs image
    a = rng.integers(0, N + 1, 4000); c = rng.integers(0, N + 1, 4000)
    p = m.comparer(zt[torch.tensor([rng.choice(it[x]) for x in a], device=dev)][:, None],
                   zt[torch.tensor([rng.choice(it[x]) for x in c], device=dev)][:, None]).argmax(1).cpu().numpy()
    res["comparaison_0_N"] = float((p == cible(a, c)).mean())
    res["comparaison_images_0_20"] = bloc(*paires(4000, 1, 1, T_MAX))
    # 3. lengths: 1-3 (seen) vs 4-6 (never seen), sum vs image (total <= 20) and sum vs sum
    res["longueurs"] = {}
    for L in [2, 3, 4, 5, 6]:
        res["longueurs"][f"{L} termes vs image"] = bloc(*paires(3000, L, 1, max(1, T_MAX // L), total_max=T_MAX))
        res["longueurs"][f"{L} termes vs 2 termes"] = bloc(*paires(3000, L, 2, max(1, T_MAX // L)))
    # 4. chained successors: [a]+[1]*k vs the image of a+k
    res["successeurs"] = {}
    for kk in [1, 2, 3, 5, 10]:
        res["successeurs"][f"+1 x{kk}"] = bloc(*paires(2000, kk + 1, 1, T_MAX - kk, total_max=T_MAX, succ=True))
    res["a+b_vs_image"] = bloc(*paires(3000, 2, 1, T_MAX, total_max=T_MAX))
    # 5. totals beyond 20: sum vs sum
    res["totaux"] = {}
    for lo, hi in [(0, 20), (21, 40), (41, 60)]:
        res["totaux"][f"{lo}-{hi}"] = bloc(*paires(3000, 3, 3, T_MAX, total_min=lo, total_max=hi))
    # 6. order and grouping: [a]+[b]+[c] vs the reversed permutation; commutativity of h
    t = rng.integers(0, 8, (2000, 3))
    p, _, _, _ = juger([list(x) for x in t], [list(x[::-1]) for x in t])
    res["ordre_abc_vs_cba_egal"] = float((p == EGAL).mean())
    a = rng.integers(0, T_MAX + 1, 2000); bb = rng.integers(0, T_MAX + 1, 2000)
    ia = zt[torch.tensor([rng.choice(it[x]) for x in a], device=dev)][:, None]; ib = zt[torch.tensor([rng.choice(it[x]) for x in bb], device=dev)][:, None]
    op = torch.zeros(2000, dtype=torch.long, device=dev)
    res["commutativite_en_pas"] = float(((h.pas(ia, ib, op) - h.pas(ib, ia, op)).abs().mean() / abs(k)).item())
    res["decalage_h_moins_somme_en_pas"] = float(((h.pas(ia, ib, op) - (ia + ib)).mean() / k).item())
    m.train(); h.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depart", type=int, choices=[3, 9], required=True)
    ap.add_argument("--variante", choices=["general", "succ"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--plafond", type=int, default=9000)
    args = ap.parse_args()
    N = args.depart; dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed); gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    nom = f"D{N}_{args.variante}_seed{args.seed}"
    out = f"results/expressions/{nom}"; os.makedirs(out, exist_ok=True)

    cfg = f"modeles/plage_N0-{N}_x1-8_z1_tete128_egal"
    best = [json.load(open(f"{cfg}/{x}")) for x in os.listdir(cfg) if x.endswith(".json")]
    best = [x for x in best if x["meilleure_graine"]][0]
    m = Comparateur(8, 1, 3, 128).to(dev); m.load_state_dict(torch.load(best["chemin"], map_location=dev))
    h = Brique(n_ops=1).to(dev)
    print(f"{nom} : départ {best['chemin']}", flush=True)

    X, _, ordre, compte, debut = charger_pool(["train_plage15", "train_plage16a20"], dev)
    Xt, yt_, _, _, _ = charger_pool(["test_plage15", "test_plage16a20"], dev)
    it = {n: np.where(yt_ == n)[0] for n in range(T_MAX + 1)}
    g = np.load("data/grand_mix.npz")
    Xg = torch.tensor(g["X"], dtype=torch.uint8, device=dev).unsqueeze(1)
    Xg = torch.cat([torch.zeros_like(Xg[:200]), Xg])
    yg = np.concatenate([np.zeros(200, int), g["y"]]); modeg = np.concatenate([np.repeat([0, 1], 100), g["mode"]])
    T, Gm = (Xt, yt_, it), (Xg, yg, modeg)

    t0 = time.time()
    m.eval()
    with torch.no_grad():
        zt0 = encoder(m, Xt)[:, 0]
    a = rng.integers(0, N + 1, 4000); c = rng.integers(0, N + 1, 4000)
    with torch.no_grad():
        p = m.comparer(zt0[torch.tensor([rng.choice(it[x]) for x in a], device=dev)][:, None],
                       zt0[torch.tensor([rng.choice(it[x]) for x in c], device=dev)][:, None]).argmax(1).cpu().numpy()
    avant = {"comparaison_0_N": float((p == cible(a, c)).mean()), "ligne": mesures_au_dela(m, N, *Gm, np.random.default_rng(1), n_min=0)}
    m.train()

    opt = torch.optim.AdamW([{"params": m.enc.parameters(), "lr": 2e-4}, {"params": m.tete.parameters(), "lr": 2e-3},
                             {"params": h.parameters(), "lr": 2e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    A, niveau, fenetre, historique, fin = N, 0, [], [], None
    for step in range(1, args.plafond + 1):
        Gl, Dl = tirer_exemples(args.bs, A, niveau, args.variante, rng)
        TG, LG = tableau(Gl, L_MAX); TD, LD = tableau(Dl, L_MAX)
        ns = torch.tensor(np.concatenate([TG.ravel(), TD.ravel()]), device=dev)
        idx = tirer(ns, ordre, compte, debut, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(augmenter(flottant(X[idx]))).float()
        ZG = z[:args.bs * L_MAX].view(args.bs, L_MAX); ZD = z[args.bs * L_MAX:].view(args.bs, L_MAX)
        OPS = torch.zeros(args.bs, L_MAX, dtype=torch.long, device=dev)
        vg = h.valeur(ZG, OPS, torch.tensor(LG, device=dev)); vd = h.valeur(ZD, OPS, torch.tensor(LD, device=dev))
        t = torch.tensor(cible(TG.sum(1), TD.sum(1)), device=dev)
        logits = m.comparer(vg, vd)
        loss = F.cross_entropy(logits.float(), t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        fenetre.append((logits.argmax(1) == t).float().mean().item()); fenetre = fenetre[-300:]
        if len(fenetre) == 300 and np.mean(fenetre) > 0.9 and fin is None:
            if niveau < len(NIVEAUX) - 1:
                niveau += 1
            elif A < T_MAX:
                A += 1
            else:
                fin = step
            historique.append({"pas": step, "A": A, "niveau": niveau}); fenetre = []
            print(f"pas {step:5d} : termes 0..{A}, écart minimal {NIVEAUX[niveau]} ({time.time() - t0:.0f} s)", flush=True)
        if step % 1000 == 0:
            print(f"pas {step:5d}  perte {loss.item():.3f}  exactitude {np.mean(fenetre or [0]):.3f}  A {A} niveau {niveau} ({time.time() - t0:.0f} s)", flush=True)
        if fin is not None and step >= fin + 500:
            break
    torch.save(m.state_dict(), f"{out}/modele.pt"); torch.save(h.state_dict(), f"{out}/h.pt")
    apres = mesurer(m, h, N, T, Gm, np.random.default_rng(1), dev)
    metrics = {"nom": nom, "depart": N, "variante": args.variante, "seed": args.seed, "modele_depart": best["chemin"],
               "pas_arret": step, "fin_curriculum": fin, "A_final": A, "niveau_final": niveau, "curriculum": historique,
               "avant": avant, "apres": apres, "duree_s": time.time() - t0}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)
    e = apres["echelle"]; lg = apres["longueurs"]; tt = apres["totaux"]
    print(f"RÉSULTAT {nom} | arrêt {step}, curriculum {fin}, A {A} | 10+10 lu {e['10+10_lu']:.1f}, « = » vs 20 g/d {e['10+10_vs_20_egal_gauche']:.2f}/{e['10+10_vs_20_egal_droite']:.2f}"
          f" | comparaison 0-{N} {avant['comparaison_0_N']:.3f} -> {apres['comparaison_0_N']:.3f}"
          f" | 5 termes vs image : = {lg['5 termes vs image'].get('egal', {}).get('exact', float('nan')):.2f} ±1 {lg['5 termes vs image'].get('ecart1', {}).get('exact', float('nan')):.2f}"
          f" | +1 x10 : = {apres['successeurs']['+1 x10'].get('egal', {}).get('exact', float('nan')):.2f}"
          f" | totaux 41-60 : = {tt['41-60'].get('egal', {}).get('exact', float('nan')):.2f} ±1 côté {tt['41-60'].get('ecart1', {}).get('cote', float('nan')):.2f}"
          f" | abc=cba {apres['ordre_abc_vs_cba_egal']:.2f}", flush=True)


if __name__ == "__main__":
    main()
