"""Addition with images: addition becomes a comparison.

s_hat = h(z_a, z_b) is a "sum" computed in latent space; the existing MLP comparison head judges s_hat against
z_c (the image of a pile of c objects): <, = or >. No number labels.

Starting model: a model from the range series (0-N with 0 = empty image, MLP head 128, x1/8, z = 1, with "equal"),
best seed.
h: ReLU MLP 2 -> 128 -> 1 (normal learning rate 2e-3); encoder at rate 2e-4; comparison head at the normal rate.
Trained on addition only. c: 1/3 "equal" (c = a + b); otherwise c = a + b +/- k with k >= minimum gap.
Curriculum (accuracy over the last 300 steps > 90%):
  1. minimum gap: 30% of the sum, then 20%, 10%, then +/-1 (at least 1 object);
  2. once at +/-1: the operand range grows by one number (a, b <= A, a + b <= 20), up to the maximum.
  End: maximum range reached and validated, then 500 consolidation steps (cap 8000 steps).
Variants: "general" (a + b) or "succ" (successor, a + 1 only).

Measurements before and after (test images: test_plage15, test_plage16a20, grand_mix for the number line):
  z line (readout of 10/15/20 with a line fitted on the original 0..N, n vs n+1 beyond N, spread);
  forgetting of comparison (accuracy on 0..N, "equal" beyond); addition per sum band for c = s, s+/-1, s+/-2;
  commutativity; a + 0 = a; measure C: h(u, v) - (u + v) compared with -b (z ~ k n + b); addition by repeated successors.

Usage (from the repository root): uv run python addition.py --depart 3 --variante general --seed 0
Writes results/addition/D<N>_<variante>_seed<s>/{metrics.json, modele.pt, h.pt}
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

S_MAX = 20
NIVEAUX = [0.3, 0.2, 0.1, 0.0]   # relative minimum gap; 0.0 = +/-1


def charger_pool(noms, dev):
    """uint8 images on the GPU, with class 0 (empty image); returns X, y, and per-number index."""
    Xs, ys = [], []
    for nom in noms:
        d = np.load(f"data/{nom}.npz"); Xs.append(d["X"]); ys.append(d["y"])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    n0 = len(X) // (y.max()) // 2
    X = np.concatenate([np.zeros((n0,) + X.shape[1:], np.uint8), X]); y = np.concatenate([np.zeros(n0, int), y])
    Xt = torch.tensor(X, dtype=torch.uint8, device=dev).unsqueeze(1)
    yt = torch.tensor(y, device=dev)
    ordre = torch.argsort(yt); compte = torch.bincount(yt, minlength=S_MAX + 1); debut = torch.cumsum(compte, 0) - compte
    return Xt, y, ordre, compte, debut


def tirer(ns, ordre, compte, debut, gen):
    return ordre[debut[ns] + (torch.rand(len(ns), device=ns.device, generator=gen) * compte[ns]).long()]


def tirer_paires(nb, A, niveau, variante, rng):
    """Operands and c according to the curriculum."""
    if variante == "succ":
        a = rng.integers(0, min(A, S_MAX - 1) + 1, nb); b = np.ones(nb, int)
    else:
        a = rng.integers(0, A + 1, nb); b = rng.integers(0, A + 1, nb)
        trop = a + b > S_MAX
        while trop.any():
            a[trop] = rng.integers(0, A + 1, trop.sum()); b[trop] = rng.integers(0, A + 1, trop.sum()); trop = a + b > S_MAX
    s = a + b
    c = s.copy()
    ineg = rng.random(nb) >= 1 / 3
    dmin = np.maximum(1, np.round(NIVEAUX[niveau] * s)).astype(int)
    k = dmin + rng.integers(0, np.maximum(1, np.round(0.3 * s)).astype(int) + 1)
    signe = np.where(rng.random(nb) < 0.5, -1, 1)
    c2 = s + signe * k
    c2 = np.where((c2 < 0) | (c2 > S_MAX), s - signe * k, c2)
    c2 = np.clip(c2, 0, S_MAX)
    c2 = np.where(c2 == s, np.where(s < S_MAX, s + 1, s - 1), c2)
    c[ineg] = c2[ineg]
    return a, b, c


class Additionneur(nn.Module):
    def __init__(self, dim_z=1, cache=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * dim_z, cache), nn.ReLU(), nn.Linear(cache, dim_z))

    def forward(self, za, zb):
        return self.f(torch.cat([za, zb], 1))


@torch.no_grad()
def mesurer(m, h, N, T, G, rng, dev):
    """T: test set (X, y, index) from 0 to 20; G: grand_mix + empty images (number line, fixed size)."""
    m.eval()
    Xt, yt, it = T
    zt = encoder(m, Xt)
    res = {}
    # 1. number line and comparison beyond N (range-series measurements)
    res["ligne"] = mesures_au_dela(m, N, *G, rng, n_min=0)
    zn = zt[:, 0].cpu().numpy()
    moy = np.array([zn[it[n]].mean() for n in range(S_MAX + 1)]); et = np.array([zn[it[n]].std() for n in range(S_MAX + 1)])
    k, b = np.polyfit(np.arange(S_MAX + 1), moy, 1)
    res["z_par_n"] = {"moyenne": moy.tolist(), "ecart_type": et.tolist(), "k": float(k), "b": float(b), "z0": float(moy[0])}
    # 2. plain comparison within 0..N
    a = rng.integers(0, N + 1, 6000); c = rng.integers(0, N + 1, 6000)
    i = np.array([rng.choice(it[x]) for x in a]); j = np.array([rng.choice(it[x]) for x in c])
    p = m.comparer(zt[i], zt[j]).argmax(1).cpu().numpy()
    res["comparaison_0_N"] = float((p == cible(a, c)).mean())
    if h is None:
        m.train(); return res
    h.eval()
    # 3. addition per sum band, c = s + d
    bandes = {"s<=N": (0, N), "N<s<=2N": (N + 1, min(2 * N, S_MAX)), "s>2N": (2 * N + 1, S_MAX)}
    add = {}
    for nb, (lo, hi) in bandes.items():
        if lo > hi:
            continue
        r_ = {}
        for d in [0, -1, 1, -2, 2]:
            s = rng.integers(lo, hi + 1, 3000); a = (rng.random(3000) * (s + 1)).astype(int); bb = s - a
            cc = s + d; ok = (cc >= 0) & (cc <= S_MAX); a, bb, s, cc = a[ok], bb[ok], s[ok], cc[ok]
            ia = np.array([rng.choice(it[x]) for x in a]); ib = np.array([rng.choice(it[x]) for x in bb])
            ic = np.array([rng.choice(it[x]) for x in cc])
            sh = h(zt[ia], zt[ib])
            p = m.comparer(sh, zt[ic]).argmax(1).cpu().numpy()
            r_[f"c=s{d:+d}" if d else "c=s"] = float((p == cible(s, cc)).mean())
        add[nb] = r_
    res["addition"] = add
    # commutativity, a + 0 = a, measure C
    a = rng.integers(0, S_MAX + 1, 4000); bb = rng.integers(0, S_MAX + 1, 4000); g = a + bb <= S_MAX; a, bb = a[g], bb[g]
    ia = np.array([rng.choice(it[x]) for x in a]); ib = np.array([rng.choice(it[x]) for x in bb])
    h1, h2 = h(zt[ia], zt[ib]), h(zt[ib], zt[ia])
    miroir = (m.comparer(h1, h2).argmax(1) == EGAL).float().mean().item()
    res["commutativite"] = {"ecart_moyen_en_pas": float((h1 - h2).abs().mean().item() / abs(k)), "egal_juge": miroir}
    ecart = (h1 - (zt[ia] + zt[ib]))[:, 0].cpu().numpy()
    res["mesure_C"] = {"h_moins_somme_en_pas": float(ecart.mean() / k), "moins_b_en_pas": float(-b / k),
                       "h_moins_somme_ecart_type_en_pas": float(ecart.std() / abs(k))}
    i0 = np.array([rng.choice(it[0]) for _ in a])
    ia2 = np.array([rng.choice(it[x]) for x in a])
    ha0 = h(zt[ia], zt[i0])
    res["a_plus_0"] = {"egal_juge": float((m.comparer(ha0, zt[ia2]).argmax(1) == EGAL).float().mean().item()),
                       "ecart_moyen_en_pas": float((ha0 - zt[ia]).abs().mean().item() / abs(k))}
    # 5. repeated successors: a + b computed as h(...h(h(z_a, z_1), z_1)..., z_1)
    succ = {}
    for nbp in [1, 2, 3, 5]:
        a = rng.integers(0, S_MAX - nbp + 1, 2000)
        ia = np.array([rng.choice(it[x]) for x in a]); s_ = zt[ia]
        for _ in range(nbp):
            i1 = np.array([rng.choice(it[1]) for _ in a]); s_ = h(s_, zt[i1])
        r_ = {}
        for d in [0, -1, 1]:
            cc = a + nbp + d; ok = (cc >= 0) & (cc <= S_MAX)
            ic = np.array([rng.choice(it[x]) for x in cc[ok]])
            p = m.comparer(s_[torch.tensor(ok, device=dev)], zt[ic]).argmax(1).cpu().numpy()
            r_[f"c=s{d:+d}" if d else "c=s"] = float((p == cible(a[ok] + nbp, cc[ok])).mean())
        succ[f"+1 x{nbp}"] = r_
    res["successeurs_repetes"] = succ
    m.train(); h.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depart", type=int, choices=[3, 9], required=True)
    ap.add_argument("--variante", choices=["general", "succ"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=384)
    ap.add_argument("--plafond", type=int, default=8000)
    args = ap.parse_args()
    N = args.depart
    dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed); gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    nom = f"D{N}_{args.variante}_seed{args.seed}"
    out = f"results/addition/{nom}"; os.makedirs(out, exist_ok=True)

    # starting model: best seed of the range series
    cfg = f"modeles/plage_N0-{N}_x1-8_z1_tete128_egal"
    meta = [json.load(open(f)) for f in sorted(os.listdir(cfg) and [f"{cfg}/{x}" for x in os.listdir(cfg) if x.endswith(".json")])]
    best = [x for x in meta if x["meilleure_graine"]][0]
    m = Comparateur(8, 1, 3, 128).to(dev); m.load_state_dict(torch.load(best["chemin"], map_location=dev))
    h = Additionneur().to(dev)
    print(f"{nom} : départ {best['chemin']}", flush=True)

    X, _, ordre, compte, debut = charger_pool(["train_plage15", "train_plage16a20"], dev)
    Xt, yt_, _, _, _ = charger_pool(["test_plage15", "test_plage16a20"], dev)
    it = {n: np.where(yt_ == n)[0] for n in range(S_MAX + 1)}
    g = np.load("data/grand_mix.npz")
    Xg = torch.tensor(g["X"], dtype=torch.uint8, device=dev).unsqueeze(1)
    Xg = torch.cat([torch.zeros_like(Xg[:200]), Xg])
    yg = np.concatenate([np.zeros(200, int), g["y"]]); modeg = np.concatenate([np.repeat([0, 1], 100), g["mode"]])
    T, G = (Xt, yt_, it), (Xg, yg, modeg)

    t0 = time.time()
    avant = mesurer(m, None, N, T, G, np.random.default_rng(1), dev)
    print(f"avant : comparaison 0-{N} {avant['comparaison_0_N']:.3f}, lit 10/15/20 "
          + " / ".join(f"{avant['ligne']['taille']['lecture'][n]:.1f}" for n in (10, 15, 20)), flush=True)

    opt = torch.optim.AdamW([{"params": m.enc.parameters(), "lr": 2e-4}, {"params": m.tete.parameters(), "lr": 2e-3},
                             {"params": h.parameters(), "lr": 2e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    A_max = S_MAX if args.variante == "general" else S_MAX - 1
    A, niveau, fenetre, historique, fin_curriculum = N, 0, [], [], None
    suivi_z = []
    for step in range(1, args.plafond + 1):
        m.train(); h.train()
        a, b, c = tirer_paires(args.bs, A, niveau, args.variante, rng)
        ia = tirer(torch.tensor(a, device=dev), ordre, compte, debut, gen)
        ib = tirer(torch.tensor(b, device=dev), ordre, compte, debut, gen)
        ic = tirer(torch.tensor(c, device=dev), ordre, compte, debut, gen)
        t = torch.tensor(cible(a + b, c), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(augmenter(flottant(torch.cat([X[ia], X[ib], X[ic]]))))
            za, zb, zc = z[:args.bs], z[args.bs:2 * args.bs], z[2 * args.bs:]
            logits = m.comparer(h(za.float(), zb.float()), zc)
        loss = F.cross_entropy(logits.float(), t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        fenetre.append((logits.argmax(1) == t).float().mean().item()); fenetre = fenetre[-300:]
        if len(fenetre) == 300 and np.mean(fenetre) > 0.9 and fin_curriculum is None:
            if niveau < len(NIVEAUX) - 1:
                niveau += 1
            elif A < A_max:
                A += 1
            else:
                fin_curriculum = step
            historique.append({"pas": step, "A": A, "niveau": niveau}); fenetre = []
            print(f"pas {step:5d} : plage des opérandes 0..{A}, écart minimal {NIVEAUX[niveau]} ({time.time() - t0:.0f} s)", flush=True)
        if step % 250 == 0:
            m.eval()
            with torch.no_grad():
                zz = encoder(m, Xt)[:, 0].cpu().numpy()
            moy = np.array([zz[it[n]].mean() for n in range(S_MAX + 1)])
            kk, bb_ = np.polyfit(np.arange(S_MAX + 1), moy, 1)
            suivi_z.append({"pas": step, "z0": float(moy[0]), "k": float(kk), "b": float(bb_), "exactitude_lot": float(np.mean(fenetre or [0]))})
            m.train()
        if step % 1000 == 0:
            print(f"pas {step:5d}  perte {loss.item():.3f}  exactitude {np.mean(fenetre or [0]):.3f}  A {A} niveau {niveau} "
                  f"({time.time() - t0:.0f} s)", flush=True)
        if fin_curriculum is not None and step >= fin_curriculum + 500:
            break
    torch.save(m.state_dict(), f"{out}/modele.pt"); torch.save(h.state_dict(), f"{out}/h.pt")
    apres = mesurer(m, h, N, T, G, np.random.default_rng(1), dev)
    metrics = {"nom": nom, "depart": N, "variante": args.variante, "seed": args.seed, "modele_depart": best["chemin"],
               "pas_arret": step, "fin_curriculum": fin_curriculum, "A_final": A, "niveau_final": niveau,
               "curriculum": historique, "suivi_z": suivi_z, "avant": avant, "apres": apres, "duree_s": time.time() - t0}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)
    ad = apres["addition"]
    print(f"RÉSULTAT {nom} | arrêt {step}, curriculum fini {fin_curriculum}, A {A} | addition c=s / s±1 par bande : "
          + " ; ".join(f"{k}: {v['c=s']:.2f} / {(v.get('c=s-1', np.nan) + v.get('c=s+1', np.nan)) / 2:.2f}" for k, v in ad.items())
          + f" | lit 10/15/20 avant " + "/".join(f"{avant['ligne']['taille']['lecture'][n]:.1f}" for n in (10, 15, 20))
          + " après " + "/".join(f"{apres['ligne']['taille']['lecture'][n]:.1f}" for n in (10, 15, 20))
          + f" | comparaison 0-{N} {avant['comparaison_0_N']:.3f} -> {apres['comparaison_0_N']:.3f}"
          + f" | C : h-(u+v) {apres['mesure_C']['h_moins_somme_en_pas']:.2f} pas, -b {apres['mesure_C']['moins_b_en_pas']:.2f}", flush=True)


if __name__ == "__main__":
    main()
