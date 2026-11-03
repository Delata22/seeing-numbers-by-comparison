"""Series 2: adding "-" to the recurrent brick (same formats as expressions.py, symbols "+" and "-").

s <- h(s, z_k, e_op), read left to right. Totals and INTERMEDIATE RESULTS are always >= 0 during training.
Starting models (--init):
  A: model from series 1, general variant (best seed): the brick already knows "+", we add e_moins ("minus");
  B: control, base model (range series): "+" and "-" learned together, fresh brick.
Same curriculum as series 1 (gap, then term range up to 20).

Key measurements: expressions with a NEGATIVE intermediate result (never seen), e.g. [2]-[5]+[4] ? [1],
by depth of the negative excursion, vs the same expressions reordered without going negative ([2]+[4]-[5]);
a - a vs the empty image; (a + b) - b vs a; forgetting of "+"; learning speed (A vs B); scale of h(., ., e_minus).

Usage (from the repository root): uv run python expressions_moins.py --depart 3 --init A --seed 0
Writes results/expressions_moins/D<N>_<A|B>_seed<s>/{metrics.json, modele.pt, h.pt}
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from run import augmenter
from compare import GAUCHE, EGAL, DROITE, cible, flottant
from reduction import Comparateur, encoder
from plage import mesures_au_dela
from addition import charger_pool, tirer
from expressions import Brique, tableau, T_MAX, L_MAX, NIVEAUX, PLUS, MOINS


def cote_aleatoire(L, A, rng):
    """Terms and symbols of one side, with intermediate results >= 0."""
    t = [int(rng.integers(0, A + 1))]; ops = [PLUS]; s = t[0]
    for _ in range(L - 1):
        op = MOINS if rng.random() < 0.5 else PLUS
        x = int(rng.integers(0, A + 1)) if op == PLUS else int(rng.integers(0, min(A, s) + 1))
        s = s + x if op == PLUS else s - x
        t.append(x); ops.append(op)
    return t, ops, s


def cote_vers(total, L, A, rng):
    """Side of L terms whose total equals 'total' (intermediates >= 0); None on failure."""
    if L == 1:
        return ([total], [PLUS]) if 0 <= total <= A else None
    for _ in range(30):
        t, ops, _ = cote_aleatoire(L - 1, A, rng)
        s = sum(x if o == PLUS else -x for x, o in zip(t, ops))
        if total >= s and total - s <= A:
            return t + [total - s], ops + [PLUS]
        if total < s and s - total <= A:
            return t + [s - total], ops + [MOINS]
    return None


def tirer_exemples(nb, A, niveau, rng):
    G, D = [], []
    while len(G) < nb:
        typ = rng.integers(4)
        Lg = 1 if typ in (0, 2) else int(rng.integers(2, L_MAX + 1))
        Ld = 1 if typ in (0, 1) else int(rng.integers(2, L_MAX + 1))
        tg, og, Tg = cote_aleatoire(Lg, A, rng)
        egal = rng.random() < 1 / 3 or (typ == 3 and rng.random() < 0.3)
        if egal:
            Td = Tg
        else:
            dmin = max(1, int(round(NIVEAUX[niveau] * max(Tg, 1))))
            if typ == 3 and rng.random() < 0.5:
                dmin = 1
            k = dmin + int(rng.integers(0, max(1, round(0.3 * max(Tg, 1))) + 1))
            Td = Tg + (k if rng.random() < 0.5 else -k)
        d = cote_vers(Td, Ld, A, rng) if Td >= 0 else None
        if d is None:
            continue
        G.append((tg, og)); D.append(d)
    return G, D


def total(t, o):
    return sum(x if op == PLUS else -x for x, op in zip(t, o))


def preparer(cotes, Lmax=None):
    T, L = tableau([c[0] for c in cotes], Lmax)
    O, _ = tableau([c[1] for c in cotes], Lmax)
    return T, O, L


@torch.no_grad()
def mesurer(m, h, N, T, G, rng, dev):
    m.eval()
    Xt, yt, it = T
    zt = encoder(m, Xt)[:, 0]
    res = {"ligne": mesures_au_dela(m, N, *G, rng, n_min=0)}
    moy = np.array([zt[torch.tensor(it[n], device=dev)].mean().item() for n in range(T_MAX + 1)])
    k, b = np.polyfit(np.arange(T_MAX + 1), moy, 1)
    lire = lambda v: (v - b) / k

    def valeur(cotes):
        T_, O, L = preparer(cotes)
        idx = np.vectorize(lambda n: rng.choice(it[n]))(T_)
        Z = zt[torch.tensor(idx, device=dev)]
        return h.valeur(Z, torch.tensor(O, device=dev), torch.tensor(L, device=dev))

    def juger(Gc, Dc):
        l = m.comparer(valeur(Gc), valeur(Dc)).float()
        tg = np.array([total(*c) for c in Gc]); td = np.array([total(*c) for c in Dc])
        p = l.argmax(1).cpu().numpy(); cote = ((l[:, GAUCHE] > l[:, DROITE]).cpu().numpy() == (tg > td))
        return p, cote, tg, td

    def bloc(Gc, Dc):
        p, cote, tg, td = juger(Gc, Dc); d = np.abs(tg - td); r = {}
        for nom, sel in [("egal", d == 0), ("ecart1", d == 1), ("ecart2-3", (d >= 2) & (d <= 3)), ("ecart>=5", d >= 5)]:
            if sel.any():
                r[nom] = {"exact": float((p[sel] == cible(tg[sel], td[sel])).mean()), "n": int(sel.sum())}
                if nom != "egal":
                    r[nom]["cote"] = float(cote[sel].mean())
        return r

    def contre_image(Gc, nb_par=1):
        """Each expression vs the image of its total + a gap drawn from {0, 0, +/-1, +/-2, +/-5}."""
        Dc = []
        for c in Gc:
            tt = total(*c) + int(rng.choice([0, 0, -1, 1, -2, 2, -5, 5]))
            tt = min(max(tt, 0), T_MAX)
            Dc.append(([tt], [PLUS]))
        return bloc(Gc, Dc)

    # 1. negative excursion: [a] - [b] + [c] with a < b, total >= 0, vs the image of the total; and reordered [a] + [c] - [b]
    res["passage_negatif"] = {}
    for nom, (lo, hi) in {"profondeur 1-2": (1, 2), "profondeur 3-5": (3, 5), "profondeur 6-10": (6, 10)}.items():
        neg, pos = [], []
        while len(neg) < 2000:
            a = int(rng.integers(0, 11)); p_ = int(rng.integers(lo, hi + 1)); bb = a + p_
            c = int(rng.integers(p_, min(T_MAX, p_ + 10) + 1))
            if bb > T_MAX or a - bb + c > T_MAX:
                continue
            neg.append(([a, bb, c], [PLUS, MOINS, PLUS])); pos.append(([a, c, bb], [PLUS, PLUS, MOINS]))
        res["passage_negatif"][nom] = {"negatif_a-b+c": contre_image(neg), "reordonne_a+c-b": contre_image(pos)}
    # control seen in training: a - b + c with a >= b
    ctl = []
    while len(ctl) < 2000:
        a = int(rng.integers(0, T_MAX + 1)); bb = int(rng.integers(0, a + 1)); c = int(rng.integers(0, T_MAX + 1))
        if a - bb + c <= T_MAX:
            ctl.append(([a, bb, c], [PLUS, MOINS, PLUS]))
    res["a-b+c_sans_passage_negatif"] = contre_image(ctl)
    # 2. a - a vs the empty image; (a + b) - b vs a
    a = rng.integers(0, T_MAX + 1, 2000)
    res["a-a_vs_0"] = bloc([([int(x), int(x)], [PLUS, MOINS]) for x in a], [([0], [PLUS])] * 2000)
    ab = [(int(x), int(rng.integers(0, T_MAX - x + 1))) for x in a]
    res["(a+b)-b_vs_a"] = bloc([([x, y, y], [PLUS, PLUS, MOINS]) for x, y in ab], [([x], [PLUS]) for x, _ in ab])
    # 3. forgetting: "+" alone and image vs image
    plus = [([int(x), int(rng.integers(0, T_MAX - x + 1))], [PLUS, PLUS]) for x in a]
    res["a+b_vs_image"] = contre_image(plus)
    moins = [([int(x), int(rng.integers(0, x + 1))], [PLUS, MOINS]) for x in a]
    res["a-b_vs_image"] = contre_image(moins)
    x = rng.integers(0, N + 1, 4000); y = rng.integers(0, N + 1, 4000)
    p, _, _, _ = juger([([int(u)], [PLUS]) for u in x], [([int(v)], [PLUS]) for v in y])
    res["comparaison_0_N"] = float((p == cible(x, y)).mean())
    # 4. scale
    v = valeur([([10, 4], [PLUS, MOINS])] * 500)
    w = valeur([([10, 10], [PLUS, PLUS])] * 500)
    res["echelle"] = {"10-4_lu": float(lire(v.mean().item())), "10+10_lu": float(lire(w.mean().item()))}
    m.train(); h.train()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depart", type=int, choices=[3, 9], required=True)
    ap.add_argument("--init", choices=["A", "B"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--plafond", type=int, default=9000)
    args = ap.parse_args()
    N = args.depart; dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed); gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    nom = f"D{N}_{args.init}_seed{args.seed}"
    out = f"results/expressions_moins/{nom}"; os.makedirs(out, exist_ok=True)

    m = Comparateur(8, 1, 3, 128).to(dev); h = Brique(n_ops=2).to(dev)
    if args.init == "A":
        cfg = f"modeles/expressions_depart0-{N}_general"
    else:
        cfg = f"modeles/plage_N0-{N}_x1-8_z1_tete128_egal"
    meta = [json.load(open(f"{cfg}/{x}")) for x in os.listdir(cfg) if x.endswith(".json")]
    best = [x for x in meta if x["meilleure_graine"]][0]
    m.load_state_dict(torch.load(best["chemin"], map_location=dev))
    if args.init == "A":
        h1 = Brique(n_ops=1).to(dev); h1.load_state_dict(torch.load(best["fichier_h"], map_location=dev))
        with torch.no_grad():
            h.f.load_state_dict(h1.f.state_dict()); h.e[PLUS] = h1.e[0]
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
    avant = mesurer(m, h, N, T, Gm, np.random.default_rng(1), dev)
    print(f"avant : a+b {avant['a+b_vs_image'].get('egal', {}).get('exact', 0):.2f}, a-b {avant['a-b_vs_image'].get('egal', {}).get('exact', 0):.2f}", flush=True)

    opt = torch.optim.AdamW([{"params": m.enc.parameters(), "lr": 2e-4}, {"params": m.tete.parameters(), "lr": 2e-3},
                             {"params": h.parameters(), "lr": 2e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200))
    A, niveau, fenetre, historique, fin = N, 0, [], [], None
    for step in range(1, args.plafond + 1):
        Gc, Dc = tirer_exemples(args.bs, A, niveau, rng)
        TG, OG, LG = preparer(Gc, L_MAX); TD, OD, LD = preparer(Dc, L_MAX)
        ns = torch.tensor(np.concatenate([TG.ravel(), TD.ravel()]), device=dev)
        idx = tirer(ns, ordre, compte, debut, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(augmenter(flottant(X[idx]))).float()
        ZG = z[:args.bs * L_MAX].view(args.bs, L_MAX); ZD = z[args.bs * L_MAX:].view(args.bs, L_MAX)
        vg = h.valeur(ZG, torch.tensor(OG, device=dev), torch.tensor(LG, device=dev))
        vd = h.valeur(ZD, torch.tensor(OD, device=dev), torch.tensor(LD, device=dev))
        t = torch.tensor(cible(np.array([total(*c) for c in Gc]), np.array([total(*c) for c in Dc])), device=dev)
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
    metrics = {"nom": nom, "depart": N, "variante": args.init, "init": args.init, "n_ops": 2, "seed": args.seed,
               "modele_depart": best["chemin"], "pas_arret": step, "fin_curriculum": fin, "A_final": A, "niveau_final": niveau,
               "curriculum": historique, "avant": avant, "apres": apres, "duree_s": time.time() - t0}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)
    pn = apres["passage_negatif"]
    g_ = lambda r, k_: r.get(k_, {}).get("exact", float("nan"))
    print(f"RÉSULTAT {nom} | arrêt {step}, curriculum {fin}, A {A} | négatif (= / ±1) "
          + " ; ".join(f"{k_.split()[1]}: {g_(v['negatif_a-b+c'], 'egal'):.2f}/{g_(v['negatif_a-b+c'], 'ecart1'):.2f} vs réordonné {g_(v['reordonne_a+c-b'], 'egal'):.2f}/{g_(v['reordonne_a+c-b'], 'ecart1'):.2f}" for k_, v in pn.items())
          + f" | a-a=0 {g_(apres['a-a_vs_0'], 'egal'):.2f} | (a+b)-b=a {g_(apres['(a+b)-b_vs_a'], 'egal'):.2f} | a+b {g_(apres['a+b_vs_image'], 'egal'):.2f} | comparaison 0-{N} {apres['comparaison_0_N']:.3f}", flush=True)


if __name__ == "__main__":
    main()
