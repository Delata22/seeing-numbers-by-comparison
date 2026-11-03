"""1-100 series: comparing quantities from 1 to 100, 192x192 images generated on the fly on the GPU.

Training uses a pre-built set (data/cent_X.npy, 300 images per number, generate_cent.py):
on-the-fly generation (0.37 s for 256 images) was too slow. Validation and test: fresh images.

Images: black background, n white shapes (disc, square, triangle, drawn at random for each object), no
overlap (2 px gap, 1 px margin). Base radius drawn per image between 2.5 and 4.5 px,
independently of n, then +/-15 % noise per object. Antialiased rendering from distance functions.
Batched placement: object by object, 128 candidate positions per image, keep the first valid one
(up to 12 tries). If an object finds no room, the image keeps the number actually placed (exact
label; the failure rate is recorded).

Training: n drawn from the distribution (uniform, 1/n, 1/n^2); 1/3 equal pairs when "equal" is used.
Network: reduction.Comparateur (width x1/4 by default: 8-8-16-16-32-32, first block at full resolution),
z = 1, head 128; at 192x192 the final map is 12x12. Early stopping on the validation loss (uniform
images 1-100), 200-step warm-up then constant rate, cap of 8000 steps. Checkpoint every 500 steps.

Usage (from the repository root): uv run python cent.py --distrib uniforme|inv|inv2 --seed 0 [--sans_egal] [--largeur 4]
Writes results/cent/<nom>/{metrics.json, modele.pt, reprise.pt}
"""
import argparse, copy, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compare import GAUCHE, EGAL, DROITE, cible

S = 192
N_MAX = 100
ECART, MARGE = 2.0, 1.0
FACT = torch.tensor([1.0, math.sqrt(math.pi) / math.sqrt(2), math.sqrt(4 * math.pi / math.sqrt(3)) / math.sqrt(3)])
# disc, square, triangle: circumradius / radius (same area pi r^2)


@torch.no_grad()
def generer(ns, gen, dev="cuda", candidats=128, essais=12):
    """ns: (B,) tensor of integers 1..100. Returns uint8 images (B,1,S,S) and the numbers actually placed."""
    B = len(ns); K = int(ns.max())
    fact = FACT.to(dev)
    r0 = torch.empty(B, device=dev).uniform_(2.5, 4.5, generator=gen)
    r = r0[:, None] * torch.empty(B, K, device=dev).uniform_(0.85, 1.15, generator=gen)
    forme = torch.randint(0, 3, (B, K), device=dev, generator=gen)
    Rg = r * fact[forme]                                   # circumradius
    ang = torch.rand(B, K, device=dev, generator=gen) * 2 * math.pi
    actif = torch.arange(K, device=dev)[None] < ns[:, None]
    C = torch.zeros(B, K, 2, device=dev)
    place = torch.zeros(B, K, dtype=torch.bool, device=dev)
    for k in range(K):
        a_placer = actif[:, k].clone()
        R = Rg[:, k]
        for _ in range(essais):
            if not a_placer.any():
                break
            lo = (R + MARGE)[:, None, None]
            P = lo + torch.rand(B, candidats, 2, device=dev, generator=gen) * (S - 2 * lo)
            if k > 0:
                d = torch.cdist(P, C[:, :k])                   # (B, cand, k)
                ok = (d >= R[:, None, None] + Rg[:, None, :k] + ECART) | ~place[:, None, :k]
                ok = ok.all(2)
            else:
                ok = torch.ones(B, candidats, dtype=torch.bool, device=dev)
            trouve = ok.any(1) & a_placer
            idx = ok.float().argmax(1)
            C[trouve, k] = P[trouve, idx[trouve]]
            place[trouve, k] = True
            a_placer &= ~trouve
    # Rendering
    yy, xx = torch.meshgrid(torch.arange(S, device=dev) + 0.5, torch.arange(S, device=dev) + 0.5, indexing="ij")
    img = torch.zeros(B, S, S, device=dev)
    for k in range(K):
        m = place[:, k]
        if not m.any():
            continue
        b = m.nonzero()[:, 0]
        dx = xx[None] - C[b, k, 0][:, None, None]; dy = yy[None] - C[b, k, 1][:, None, None]
        f = forme[b, k]; Rk = Rg[b, k]; t0 = ang[b, k]
        d_disque = torch.sqrt(dx * dx + dy * dy) - r[b, k][:, None, None]
        nb = torch.where(f == 1, 4, 3).float()
        # regular polygon (square or triangle): max of projections on the side normals - apothem
        d_poly = torch.full_like(dx, -1e9)
        for i in range(4):
            th = t0 + math.pi / nb + 2 * math.pi * i / nb
            proj = dx * torch.cos(th)[:, None, None] + dy * torch.sin(th)[:, None, None]
            utile = (i < nb)[:, None, None]
            d_poly = torch.where(utile, torch.maximum(d_poly, proj), d_poly)
        d_poly = d_poly - (Rk * torch.cos(math.pi / nb))[:, None, None]
        d = torch.where((f == 0)[:, None, None], d_disque, d_poly)
        img[b] = torch.maximum(img[b], (0.5 - d).clamp(0, 1))
    return (img * 255).round().to(torch.uint8)[:, None], place.sum(1)


def poids_distrib(nom):
    n = np.arange(1, N_MAX + 1, dtype=float)
    p = {"uniforme": np.ones_like(n), "inv": 1 / n, "inv2": 1 / n ** 2}[nom]
    return p / p.sum()


@torch.no_grad()
def encoder(m, X, bs=100):
    return torch.cat([m.enc(X[i:i + bs].float() / 255.0).float() for i in range(0, len(X), bs)])


def jeu_fixe(n_par, gen, dev):
    """Uniform images: n_par images per number from 1 to 100 (generated in batches)."""
    ns = torch.arange(1, N_MAX + 1, device=dev).repeat_interleave(n_par)
    Xs, ys = [], []
    for i in range(0, len(ns), 500):
        X, y = generer(ns[i:i + 500], gen, dev); Xs.append(X); ys.append(y)
    return torch.cat(Xs), torch.cat(ys).cpu().numpy()


@torch.no_grad()
def mesures(m, X, y, rng):
    m.eval()
    z = encoder(m, X)
    zn = z.cpu().numpy()[:, 0] if z.shape[1] == 1 else z.cpu().numpy()
    idx = {n: np.where(y == n)[0] for n in range(1, N_MAX + 1)}
    idx = {n: v for n, v in idx.items() if len(v) > 1}
    res = {}

    def juger(a, b):
        i = np.array([rng.choice(idx[x]) for x in a]); j = np.array([rng.choice(idx[x]) for x in b])
        j = np.where(i == j, np.array([rng.choice(idx[x]) for x in b]), j)
        l = m.comparer(z[i], z[j]).float()
        p = l.argmax(1).cpu().numpy()
        cote = ((l[:, GAUCHE] > l[:, DROITE]).cpu().numpy() == (a > b))
        return p, cote

    bandes = {"1-9": (1, 9), "10-20": (10, 20), "21-40": (21, 40), "41-70": (41, 70), "71-100": (71, 100)}
    ns = np.array(sorted(idx))
    for nb, (lo, hi) in bandes.items():
        dans = ns[(ns >= lo) & (ns <= hi)]
        a = rng.choice(dans, 4000); b = rng.choice(dans, 4000); g = a != b; a, b = a[g], b[g]
        p, cote = juger(a, b)
        r_ = {"exact_inegales": float((p == cible(a, b)).mean()), "bon_cote": float(cote.mean())}
        for dlt, cle in [(0, "n_n"), (1, "n_n1"), (2, "n_n2")]:
            aa = rng.choice(dans[dans + dlt <= hi] if dlt else dans, 2000); bb = aa + dlt
            ok = np.isin(bb, ns); aa, bb = aa[ok], bb[ok]
            p2, c2 = juger(aa, bb)
            r_[f"egal_{cle}"] = float((p2 == EGAL).mean())
            if dlt:
                r_[f"bon_cote_{cle}"] = float(c2.mean())
        res[nb] = r_
    # by min/max ratio (all sizes)
    a = rng.choice(ns, 20000); b = rng.choice(ns, 20000); g = a != b; a, b = a[g], b[g]
    p, cote = juger(a, b)
    rap = np.minimum(a, b) / np.maximum(a, b)
    res["rapport"] = {}
    for nom, (lo, hi) in {"<=0.8": (0, 0.8), "0.8-0.9": (0.8, 0.9), "0.9-0.95": (0.9, 0.95), ">0.95": (0.95, 1.0)}.items():
        s = (rap > lo) & (rap <= hi)
        if s.any():
            res["rapport"][nom] = {"exact": float((p[s] == cible(a[s], b[s])).mean()), "bon_cote": float(cote[s].mean()),
                                   "n": int(s.sum())}
    # line in z (one-dimensional z): linear versus logarithmic, spread per n
    if zn.ndim == 1:
        moy = np.array([zn[idx[n]].mean() for n in ns]); et = np.array([zn[idx[n]].std() for n in ns])
        def r2(x):
            A = np.vstack([x, np.ones_like(x)]).T
            c, *_ = np.linalg.lstsq(A, zn[np.concatenate([idx[n] for n in ns])], rcond=None)
            yy = zn[np.concatenate([idx[n] for n in ns])]
            return float(1 - ((A @ c - yy) ** 2).sum() / ((yy - yy.mean()) ** 2).sum())
        xs = np.concatenate([np.full(len(idx[n]), n, float) for n in ns])
        res["ligne"] = {"r2_lineaire": r2(xs), "r2_log": r2(np.log(xs)),
                        "pente_etalement_loglog": float(np.polyfit(np.log(ns), np.log(et + 1e-12), 1)[0]),
                        "moyenne_par_n": moy.tolist(), "ecart_type_par_n": et.tolist()}
    m.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distrib", choices=["uniforme", "inv", "inv2"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sans_egal", action="store_true")
    ap.add_argument("--largeur", type=int, default=4)
    ap.add_argument("--dim_z", type=int, default=1)
    ap.add_argument("--tete", type=int, default=128)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--plafond", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--clip", type=float, default=0.0,
                    help="gradient norm clipping (0 = none); appends _clip<x>_lr<lr> to the run name")
    ap.add_argument("--critere", choices=["perte", "exactitude"], default="exactitude",
                    help="validation measure that selects the best checkpoint and triggers stopping")
    ap.add_argument("--evaluer_reprise", action="store_true",
                    help="no training: evaluate the best checkpoint stored in reprise.pt (interrupted run)")
    args = ap.parse_args()
    from reduction import Comparateur
    dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    nom = f"{args.distrib}_L{args.largeur}_z{args.dim_z}_t{args.tete}_{'se' if args.sans_egal else 'eg'}_seed{args.seed}"
    if args.clip > 0:
        nom += f"_clip{args.clip:g}_lr{args.lr:g}"
    out = f"results/cent/{nom}"; os.makedirs(out, exist_ok=True)
    gen = torch.Generator(device=dev); gen.manual_seed(10_000 + args.seed)
    gval = torch.Generator(device=dev); gval.manual_seed(424242)   # same validation images for every run
    p = poids_distrib(args.distrib)

    Xv, yv = jeu_fixe(20, gval, dev)        # validation: 20 images per number, uniform
    rv = np.random.default_rng(500)
    idxv = {n: np.where(yv == n)[0] for n in range(1, N_MAX + 1)}
    idxv = {n: v for n, v in idxv.items() if len(v) > 1}
    nsv = np.array(sorted(idxv))
    a = rv.choice(nsv, 6000); b = rv.choice(nsv, 6000)
    if not args.sans_egal:
        b[:2000] = a[:2000]
    else:
        g = a != b; a, b = a[g], b[g]
    iv = np.array([rv.choice(idxv[x]) for x in a]); jv = np.array([rv.choice(idxv[x]) for x in b])
    jv = np.where(iv == jv, np.array([rv.choice(idxv[x]) for x in b]), jv)
    iv, jv = torch.tensor(iv, device=dev), torch.tensor(jv, device=dev)
    tv = torch.tensor(cible(yv[iv.cpu().numpy()], yv[jv.cpu().numpy()]), device=dev)

    m = Comparateur(args.largeur, args.dim_z, 2 if args.sans_egal else 3, args.tete)
    # 192x192: final map 12x12 (instead of 4x4 at 64x64)
    m.enc.proj = nn.Linear(m.enc.proj.in_features // 16 * (S // 16) ** 2, args.dim_z)
    m = m.to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    etat_suivi = {"suivi": [], "meilleur": float("inf"), "meilleur_pas": 0, "sans_progres": 0, "echecs": 0, "images": 0}
    etat, debut = None, 1
    if os.path.exists(f"{out}/reprise.pt"):
        ck = torch.load(f"{out}/reprise.pt", map_location=dev, weights_only=False)
        m.load_state_dict(ck["modele"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        etat, etat_suivi, debut = ck["meilleur_etat"], ck["suivi"], ck["pas"] + 1
        gen.set_state(ck["gen"].to("cpu") if hasattr(ck["gen"], "to") else ck["gen"])
        print(f"reprise au pas {debut}", flush=True)
    if args.evaluer_reprise:
        args.plafond = debut - 1
    print(f"{nom} : encodeur {sum(q.numel() for q in m.enc.parameters())} paramètres", flush=True)
    pt = torch.tensor(p, device=dev, dtype=torch.float)
    # Pre-built set (generate_cent.py): 300 images per number, sampled according to the distribution
    PX = torch.from_numpy(np.load("data/cent_X.npy")).to(dev)
    Py = torch.from_numpy(np.load("data/cent_y.npy")).to(dev)
    ordre = torch.argsort(Py)
    compte = torch.bincount(Py, minlength=N_MAX + 1)
    debut_n = torch.cumsum(compte, 0) - compte
    n_eg = 0 if args.sans_egal else args.bs // 3
    t0 = time.time()
    step = debut - 1
    for step in range(debut, args.plafond + 1):
        m.train()
        na = torch.multinomial(pt, args.bs, replacement=True, generator=gen) + 1
        nb_ = torch.multinomial(pt, args.bs, replacement=True, generator=gen) + 1
        if n_eg:
            nb_[:n_eg] = na[:n_eg]
        nn_ = torch.cat([na, nb_])
        ii = ordre[debut_n[nn_] + (torch.rand(len(nn_), device=dev, generator=gen) * compte[nn_]).long()]
        X, reel = PX[ii][:, None], Py[ii]
        ra, rb = reel[:args.bs].cpu().numpy(), reel[args.bs:].cpu().numpy()
        t = torch.tensor(cible(ra, rb), device=dev)
        k = np.random.randint(4)
        x = torch.rot90(X.float() / 255.0, k, (2, 3))
        if np.random.rand() < 0.5:
            x = x.flip(3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(x)
            logits = m.comparer(z[:args.bs], z[args.bs:])
        loss = F.cross_entropy(logits.float(), t)
        opt.zero_grad(); loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(m.parameters(), args.clip)
        opt.step(); sched.step()
        if step % 100 == 0:
            m.eval()
            with torch.no_grad():
                zv = encoder(m, Xv)
                l = m.comparer(zv[iv], zv[jv]).float()
                pl = F.cross_entropy(l, tv).item(); pa = (l.argmax(1) == tv).float().mean().item()
            es = etat_suivi
            es["suivi"].append({"pas": step, "perte_val": pl, "acc_val": pa, "s": time.time() - t0})
            if args.critere == "perte":
                score = pl; seuil = 0.01 * es["meilleur"] if es["meilleur"] >= 1e-3 else 5e-4
            else:  # accuracy: minimize 1 - accuracy, progress = +0.1 point
                score = 1 - pa; seuil = 1e-3
            if es["meilleur"] - score > seuil or es["meilleur"] == float("inf"):
                es["meilleur"], es["meilleur_pas"], es["sans_progres"] = score, step, 0
                etat = copy.deepcopy(m.state_dict())
            else:
                es["sans_progres"] += 1
            print(f"pas {step:5d}  perte val {pl:.4f}  exactitude val {pa:.4f}  ({time.time() - t0:.0f} s)", flush=True)
            if step % 500 == 0:
                torch.save({"modele": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "meilleur_etat": etat, "suivi": etat_suivi, "pas": step, "gen": gen.get_state()},
                           f"{out}/reprise.pt")
            if es["sans_progres"] >= args.patience:
                break
    m.load_state_dict(etat)
    torch.save(m.state_dict(), f"{out}/modele.pt")
    gt = torch.Generator(device=dev); gt.manual_seed(777)
    Xt, yt = jeu_fixe(40, gt, dev)
    res = mesures(m, Xt, yt, np.random.default_rng(900 + args.seed))
    es = etat_suivi
    metrics = {"nom": nom, "distrib": args.distrib, "sans_egal": args.sans_egal, "seed": args.seed,
               "largeur": args.largeur, "dim_z": args.dim_z, "tete": args.tete, "pas_arret": step,
               "meilleur_pas": es["meilleur_pas"], "critere": args.critere, "lr": args.lr, "clip": args.clip, "perte_val_min": min(x["perte_val"] for x in es["suivi"]), "suivi": es["suivi"],
               "interrompu": bool(args.evaluer_reprise), "mesures": res}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)
    print("RÉSULTAT", nom, "| arrêt", step, "|", " ".join(
        f"{b}: {res[b]['exact_inegales']:.3f}/{res[b]['bon_cote_n_n1']:.2f}" for b in ["1-9", "10-20", "21-40", "41-70", "71-100"]),
        "| R² lin/log", f"{res['ligne']['r2_lineaire']:.3f}/{res['ligne']['r2_log']:.3f}", flush=True)


if __name__ == "__main__":
    main()
