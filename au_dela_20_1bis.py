"""Test 1bis of "above 20": MIXED fine-tuning (image comparisons 0..30 + the original expressions), with encoder,
head AND brick h trained together. Question: do images with 21 to 30 objects, never compared with a sum during
training, land on the sum line?

Starting point: saved expression models (modeles/expressions_depart0-<N>_<variante>/seed<s>.pt and seed<s>_h.pt).
Each batch: half (a), half (b).
  (a) image vs image comparison, n = 0..30 (0..20: train_plage15 + train_plage16a20; 21..30: train_21a30_aire),
      1/3 equal pairs, as in part B;
  (b) expressions as in the original series (expressions.tirer_exemples), at the last curriculum stage
      (terms 0..20, minimum difference +/-1), 1 to 3 terms per side, single-term sides included; variant general or succ.
  Strict rule: no image with more than 20 objects is ever compared with a side that contains a "+" (all expression
  terms are <= 20; images 21..30 appear only in (a)).
Learning rates: encoder 2e-4, head 2e-3, h 2e-3 (those of the expressions series). Loss = mean of the two cross-entropies.
Part B measurement and criterion, every 50 steps (reading of 21..30 within +/-0.5 and n vs n+1 > 95 %); stop 500 steps
after the first step that meets it; cap of 6000 steps.
Full measurements before and after: au_dela_20_test1.evaluer (image = sum, +/-1, line offset, reading, n/n+1,
chained successors for succ, forgetting of sum vs sum on 0-60 and 61-100).

Long continuation (step 4): --depuis results/au_dela_20_1bis --sortie results/au_dela_20_1bis_long --plafond 9000 --sans_arret
--journal 1500: restarts from the already fine-tuned model (and h), same batch, same learning rates, same rule; fresh
optimizer (learning-rate warm-up over 200 steps); no stop at the criterion; every 1500 steps, reading of the images and
of the sums at n = 25, 28, 30, and image = sum at those n.

Usage (from the repository root): uv run python au_dela_20_1bis.py --depart 3 --variante general --seed 0
Writes results/au_dela_20_1bis/D<N>_<variante>_seed<s>.{pt,_h.pt,json}
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from run import augmenter
from compare import GAUCHE, DROITE, cible, flottant
from reduction import encoder
from addition import tirer
from expressions import tirer_exemples, tableau, L_MAX, NIVEAUX
from au_dela_20_B import pool
import au_dela_20_test1 as T1

dev = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depart", type=int, choices=[3, 9], required=True)
    ap.add_argument("--variante", choices=["general", "succ"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plafond", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--depuis", default=None, help="directory of already fine-tuned models (long continuation)")
    ap.add_argument("--sortie", default="results/au_dela_20_1bis")
    ap.add_argument("--sans_arret", action="store_true")
    ap.add_argument("--journal", type=int, default=0, help="every k steps: readings and image = sum at n = 25, 28, 30")
    args = ap.parse_args()
    N = args.depart; v = args.variante
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    nom = f"D{N}_{v}_seed{args.seed}"
    out = args.sortie; os.makedirs(out, exist_ok=True)
    src = f"{args.depuis}/{nom}" if args.depuis else f"modeles/expressions_depart0-{N}_{v}/seed{args.seed}"
    m, h = T1.charger(f"{src}.pt", f"{src}_h.pt")
    if args.depuis:   # draws different from those of the first fine-tuning
        rng = np.random.default_rng(1000 + args.seed); gen.manual_seed(1000 + args.seed)
    succ = v == "succ"
    avant = T1.evaluer(m, h, np.random.default_rng(7), succ=succ, oubli=True)

    X, ordre, compte, debut = pool(["train_plage15", "train_plage16a20", "train_21a30_aire"])
    ig = T1.ig; Xg = T1.Xg
    rr = np.random.default_rng(99)
    paires_test = {n: (torch.tensor(rr.choice(ig[n], 200), device=dev), torch.tensor(rr.choice(ig[n + 1], 200), device=dev)) for n in range(21, 30)}

    @torch.no_grad()
    def mesure():   # identical to part B
        m.eval()
        _, k, b = T1.ligne(m)
        zg = encoder(m, Xg)[:, 0]
        lect = {n: float((zg[torch.tensor(ig[n], device=dev)].mean().item() - b) / k) for n in range(21, 31)}
        cote = {}
        for n, (i, j) in paires_test.items():
            l1 = m.comparer(zg[i][:, None], zg[j][:, None]).float(); l2 = m.comparer(zg[j][:, None], zg[i][:, None]).float()
            cote[n] = float(((l1[:, DROITE] > l1[:, GAUCHE]).float().mean() + (l2[:, GAUCHE] > l2[:, DROITE]).float().mean()).item() / 2)
        m.train()
        ok = all(abs(lect[n] - n) <= 0.5 for n in lect) and all(c > 0.95 for c in cote.values())
        return {"lecture": lect, "cote_n_n1": cote, "ok": ok,
                "ecart_lecture_max": float(max(abs(lect[n] - n) for n in lect)), "cote_min": float(min(cote.values()))}

    @torch.no_grad()
    def journal():
        m.eval(); h.eval()
        z, k, b = T1.ligne(m); zg = encoder(m, Xg)[:, 0]; r_ = np.random.default_rng(3); j = {}
        for n in (25, 28, 30):
            zi = zg[torch.tensor(r_.choice(ig[n], T1.NB), device=dev)][:, None]
            s = T1.valeur(h, z, [T1.decomposer_sur(n, 2, r_) for _ in range(T1.NB)], r_)
            e_, _ = T1.comparer_deux_ordres(m, zi, s, np.full(T1.NB, n), np.full(T1.NB, n))
            j[str(n)] = {"lecture_image": float((zg[torch.tensor(ig[n], device=dev)].mean().item() - b) / k),
                         "lecture_somme": float((s.mean().item() - b) / k), "image_egal_somme": e_}
        m.train(); h.train()
        return j

    m.train(); h.train()
    journaux = {}
    if args.journal:
        journaux[0] = journal()
    opt = torch.optim.AdamW([{"params": m.enc.parameters(), "lr": 2e-4}, {"params": m.tete.parameters(), "lr": 2e-3},
                             {"params": h.parameters(), "lr": 2e-3}], weight_decay=1e-4)
    # long continuation: the optimizer state was not saved; gradual learning-rate warm-up over 200 steps (as in the expressions series)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, (lambda s: min(1.0, (s + 1) / 200)) if args.depuis else (lambda s: 1.0))
    ba = args.bs // 2; be = args.bs - ba
    suivi = {0: mesure()}; premier_ok = 0 if suivi[0]["ok"] else None
    exact_expr = []
    t0 = time.time()
    for step in range(1, args.plafond + 1):
        # (a) image vs image, 0..30
        a = rng.integers(0, 31, ba); b = rng.integers(0, 31, ba); b[: ba // 3] = a[: ba // 3]
        # (b) original expressions, terms 0..20, last stage (+/-1)
        Gl, Dl = tirer_exemples(be, 20, len(NIVEAUX) - 1, v, rng)
        TG, LG = tableau(Gl, L_MAX); TD, LD = tableau(Dl, L_MAX)
        assert TG.max() <= 20 and TD.max() <= 20
        ns = torch.tensor(np.concatenate([a, b, TG.ravel(), TD.ravel()]), device=dev)
        idx = tirer(ns, ordre, compte, debut, gen)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(augmenter(flottant(X[idx]))).float()
        za, zb = z[:ba], z[ba:2 * ba]
        ze = z[2 * ba:]
        ZG = ze[:be * L_MAX].view(be, L_MAX); ZD = ze[be * L_MAX:].view(be, L_MAX)
        OPS = torch.zeros(be, L_MAX, dtype=torch.long, device=dev)
        vg = h.valeur(ZG, OPS, torch.tensor(LG, device=dev)); vd = h.valeur(ZD, OPS, torch.tensor(LD, device=dev))
        la = m.comparer(za, zb); le = m.comparer(vg, vd)
        te = torch.tensor(cible(TG.sum(1), TD.sum(1)), device=dev)
        loss = 0.5 * F.cross_entropy(la.float(), torch.tensor(cible(a, b), device=dev)) + 0.5 * F.cross_entropy(le.float(), te)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        exact_expr.append((le.argmax(1) == te).float().mean().item())
        if step % 50 == 0:
            suivi[step] = mesure(); suivi[step]["exact_expressions_50"] = float(np.mean(exact_expr[-50:]))
            if suivi[step]["ok"] and premier_ok is None:
                premier_ok = step
            if step % 500 == 0:
                s_ = suivi[step]
                print(f"pas {step:5d} | écart lecture max {s_['ecart_lecture_max']:.2f}, côté min {s_['cote_min']:.2f}, "
                      f"expressions {s_['exact_expressions_50']:.2f} ({time.time() - t0:.0f} s)", flush=True)
            if premier_ok is not None and step >= premier_ok + 500 and not args.sans_arret:
                break
        if args.journal and step % args.journal == 0:
            journaux[step] = journal()
            print(f"pas {step:5d} | " + " ; ".join(f"n={n} image {x['lecture_image']:.1f} somme {x['lecture_somme']:.1f} = {x['image_egal_somme']:.2f}" for n, x in journaux[step].items()), flush=True)
    torch.save(m.state_dict(), f"{out}/{nom}.pt"); torch.save(h.state_dict(), f"{out}/{nom}_h.pt")
    apres = T1.evaluer(m, h, np.random.default_rng(7), succ=succ, oubli=True)
    json.dump({"nom": nom, "depart": N, "variante": v, "seed": args.seed, "modele_depart": f"{src}.pt", "h_depart": f"{src}_h.pt",
               "premier_pas_ok": premier_ok, "pas_arret": step, "duree_s": time.time() - t0,
               "suivi": {str(k): x for k, x in suivi.items()}, "journal": {str(k): x for k, x in journaux.items()}, "avant": avant, "apres": apres},
              open(f"{out}/{nom}.json", "w"), indent=1)
    eg = lambda r, n: r["par_n"][str(n)].get("img_somme_egal", float("nan"))
    print(f"RÉSULTAT {nom} | premier pas ok {premier_ok}, arrêt {step} | image = somme 21/25/30 avant "
          f"{eg(avant, 21):.2f}/{eg(avant, 25):.2f}/{eg(avant, 30):.2f} après {eg(apres, 21):.2f}/{eg(apres, 25):.2f}/{eg(apres, 30):.2f}"
          f" | dans la plage 20 : {avant['dans_plage']['20']['img_somme_egal']:.2f} -> {apres['dans_plage']['20']['img_somme_egal']:.2f}"
          f" | somme/somme 61-100 = : {avant['oubli']['61-100']['egal']:.2f} -> {apres['oubli']['61-100']['egal']:.2f}", flush=True)


if __name__ == "__main__":
    main()
