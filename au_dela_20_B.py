"""Part B of "above 20": does the sum help the model learn faster to read images with 21 to 30 objects?

Fine-tuning by plain comparison (image vs image, 1/3 equal pairs) on images from 0 to 30: 0..20 as usual
(data/train_plage15 + train_plage16a20, both modes), 21..30 with fixed area (data/train_21a30_aire, new).
Starting models: "addition" (expressions series 1, general variant, best seed), "succ" (successor variant, best
seed), "base" (starting range comparison model, no addition, seed 2). Starting range 0-3 or 0-9.
"controle": the "base" model, first trained by plain comparison on 0..20 for as many steps as the expression
training of the matching "addition" model (its pas_arret), same learning rates; then the fine-tuning.
The fine-tuned model is saved (results/au_dela_20_B/<name>.pt).
Learning rates: encoder 2e-4, head 2e-3 (as during addition training). Batches of 256 pairs, cap of 3000 steps.
Measured every 50 steps on the test images (grand_mix, fixed area, 21..30): reading of each n on the line fitted
on the test images 0..20 (extrapolated), and correct side for n vs n+1 (21..29).
Criterion: all readings from 21 to 30 within +/-0.5 and every n vs n+1 > 95 %; the first step that meets it is recorded.

Usage (from the repository root): uv run python au_dela_20_B.py --depart 3 --type addition --seed 0
Writes results/au_dela_20_B/D<N>_<type>_seed<s>.json
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

from run import augmenter
from compare import GAUCHE, EGAL, DROITE, cible, flottant
from reduction import Comparateur, encoder
from addition import tirer

dev = "cuda"


def pool(noms):
    Xs, ys = [], []
    for n in noms:
        d = np.load(f"data/{n}.npz"); Xs.append(d["X"]); ys.append(d["y"])
    X = np.concatenate(Xs); y = np.concatenate(ys)
    n0 = 3000
    X = np.concatenate([np.zeros((n0,) + X.shape[1:], np.uint8), X]); y = np.concatenate([np.zeros(n0, int), y])
    Xt = torch.tensor(X, dtype=torch.uint8, device=dev).unsqueeze(1); yt = torch.tensor(y, device=dev)
    ordre = torch.argsort(yt); compte = torch.bincount(yt, minlength=31); debut = torch.cumsum(compte, 0) - compte
    return Xt, ordre, compte, debut


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depart", type=int, choices=[3, 9], required=True)
    ap.add_argument("--type", choices=["addition", "succ", "base", "controle"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plafond", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=256)
    args = ap.parse_args()
    N = args.depart
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed); gen = torch.Generator(device=dev); gen.manual_seed(args.seed)
    cfg = {"addition": f"modeles/expressions_depart0-{N}_general", "succ": f"modeles/expressions_depart0-{N}_succ",
           "base": f"modeles/plage_N0-{N}_x1-8_z1_tete128_egal", "controle": f"modeles/plage_N0-{N}_x1-8_z1_tete128_egal"}[args.type]
    meta = [json.load(open(f"{cfg}/{x}")) for x in os.listdir(cfg) if x.endswith(".json")]
    best = [x for x in meta if x["meilleure_graine"]][0]
    m = Comparateur(8, 1, 3, 128).to(dev); m.load_state_dict(torch.load(best["chemin"], map_location=dev))
    X, ordre, compte, debut = pool(["train_plage15", "train_plage16a20", "train_21a30_aire"])
    # test: 0..20 (test_plage15 + test_plage16a20 + empty images) for the line; grand_mix fixed area 21..30
    from addition import charger_pool
    Xt, yt, _, _, _ = charger_pool(["test_plage15", "test_plage16a20"], dev)
    it = {n: torch.tensor(np.where(yt == n)[0], device=dev) for n in range(21)}
    g = np.load("data/grand_mix.npz")
    Xg = torch.tensor(g["X"], dtype=torch.uint8, device=dev).unsqueeze(1)
    ig = {n: np.where((g["y"] == n) & (g["mode"] == 0))[0] for n in range(21, 31)}
    rr = np.random.default_rng(99)
    paires_test = {n: (torch.tensor(rr.choice(ig[n], 200), device=dev), torch.tensor(rr.choice(ig[n + 1], 200), device=dev)) for n in range(21, 30)}

    @torch.no_grad()
    def mesure():
        m.eval()
        z = encoder(m, Xt)[:, 0]; zg = encoder(m, Xg)[:, 0]
        moy = np.array([z[it[n]].mean().item() for n in range(21)])
        k, b = np.polyfit(np.arange(21), moy, 1)
        lect = {n: float((zg[torch.tensor(ig[n], device=dev)].mean().item() - b) / k) for n in range(21, 31)}
        cote = {}
        for n, (i, j) in paires_test.items():
            l1 = m.comparer(zg[i][:, None], zg[j][:, None]).float(); l2 = m.comparer(zg[j][:, None], zg[i][:, None]).float()
            cote[n] = float(((l1[:, DROITE] > l1[:, GAUCHE]).float().mean() + (l2[:, GAUCHE] > l2[:, DROITE]).float().mean()).item() / 2)
        m.train()
        ok = all(abs(lect[n] - n) <= 0.5 for n in lect) and all(c > 0.95 for c in cote.values())
        return {"lecture": lect, "cote_n_n1": cote, "ok": ok,
                "ecart_lecture_max": float(max(abs(lect[n] - n) for n in lect)), "cote_min": float(min(cote.values()))}

    opt = torch.optim.AdamW([{"params": m.enc.parameters(), "lr": 2e-4}, {"params": m.tete.parameters(), "lr": 2e-3}], weight_decay=1e-4)
    pas_controle = 0
    if args.type == "controle":   # control phase: plain comparison on 0..20, as many steps as the expression training
        ref = [json.load(open(f"modeles/expressions_depart0-{N}_general/{x}")) for x in os.listdir(f"modeles/expressions_depart0-{N}_general") if x.endswith(".json")]
        pas_controle = [x for x in ref if x["meilleure_graine"]][0]["pas_arret"]
        for step in range(1, pas_controle + 1):
            a = rng.integers(0, 21, args.bs); b = rng.integers(0, 21, args.bs); b[: args.bs // 3] = a[: args.bs // 3]
            ia = tirer(torch.tensor(a, device=dev), ordre, compte, debut, gen); ib = tirer(torch.tensor(b, device=dev), ordre, compte, debut, gen)
            t = torch.tensor(cible(a, b), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                z = m.enc(augmenter(flottant(torch.cat([X[ia], X[ib]]))))
                logits = m.comparer(z[:args.bs], z[args.bs:])
            loss = F.cross_entropy(logits.float(), t)
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"phase de contrôle : {pas_controle} pas sur 0..20", flush=True)
    suivi = {0: mesure()}
    premier_ok = 0 if suivi[0]["ok"] else None
    t0 = time.time()
    for step in range(1, args.plafond + 1):
        a = rng.integers(0, 31, args.bs); b = rng.integers(0, 31, args.bs); b[: args.bs // 3] = a[: args.bs // 3]
        ia = tirer(torch.tensor(a, device=dev), ordre, compte, debut, gen); ib = tirer(torch.tensor(b, device=dev), ordre, compte, debut, gen)
        t = torch.tensor(cible(a, b), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(augmenter(flottant(torch.cat([X[ia], X[ib]]))))
            logits = m.comparer(z[:args.bs], z[args.bs:])
        loss = F.cross_entropy(logits.float(), t)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            suivi[step] = mesure()
            if suivi[step]["ok"] and premier_ok is None:
                premier_ok = step
            if premier_ok is not None and step >= premier_ok + 500:
                break
    os.makedirs("results/au_dela_20_B", exist_ok=True)
    nom = f"D{N}_{args.type}_seed{args.seed}"
    torch.save(m.state_dict(), f"results/au_dela_20_B/{nom}.pt")
    json.dump({"nom": nom, "depart": N, "type": args.type, "seed": args.seed, "modele_depart": best["chemin"], "pas_controle": pas_controle,
               "premier_pas_ok": premier_ok, "pas_arret": step, "duree_s": time.time() - t0,
               "suivi": {str(k): v for k, v in suivi.items()}}, open(f"results/au_dela_20_B/{nom}.json", "w"), indent=1)
    s0 = suivi[0]
    print(f"RÉSULTAT {nom} | avant : écart lecture max {s0['ecart_lecture_max']:.1f}, côté min {s0['cote_min']:.2f} | "
          f"premier pas ok : {premier_ok} | fin : écart {suivi[max(suivi)]['ecart_lecture_max']:.2f}, côté min {suivi[max(suivi)]['cote_min']:.2f}", flush=True)


if __name__ == "__main__":
    main()
