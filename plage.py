"""Training range 1 to N (N from 3 to 15): how far does the number line extend beyond N?

Same network and same training as reduction.py (constant rate after a 200-step warm-up, early
stopping on the validation loss, cap of 4000). Differences:
  - images data/train_plage15.npz restricted to 1..N (6000 per number), validation test_plage15 (1..N);
  - no held-out pair: all pairs from 1 to N are seen (1/3 equal pairs when the "equal" class is used);
  - measurements beyond N on data/grand_mix.npz (fixed size up to 20, fixed area up to 30):
      readout of the z -> n line fitted on 1..N; correct side per band (N+1..2N, beyond 2N);
      correct side and "equal" rate for n versus n+1; "equal" rate for n versus n (two images);
      indicator (range of the number line, "portee"): largest n such that, for every m from N+1 to n,
      the readout of m is within +/-0.5 and the correct side of (m-1, m) exceeds 95 % (fixed size).
      Extrapolation factor = n / N.

Option --zero: the number 0 (blank image) is added to training and validation (range 0 to N),
and the z -> n line is then fitted on 0..N (blank images added to the measurement images).

Usage (from the repository root): uv run python plage.py --n_max 5 --largeur 8 --dim_z 1 --tete 8 --seed 0 [--sans_egal]
Writes results/plage/N<N>_L<l>_z<z>_t<t>_<eg|se>_seed<s>/{metrics.json, modele.pt}
"""
import argparse, copy, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from run import augmenter
from compare import GAUCHE, EGAL, DROITE, cible, charger, flottant
from reduction import Comparateur, encoder, PATIENCE, EVAL_TOUS, PLAFOND, MONTEE

PAS_GENERALISATION = {200, 500, 1000}


class CoucheSansNorme(nn.Module):
    """Transformer layer without normalization: x + attention(x), then x + feed-forward(x)."""
    def __init__(self, d, tetes):
        super().__init__()
        self.att = nn.MultiheadAttention(d, tetes, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.att(x, x, x, need_weights=False)[0]
        return x + self.ff(x)


class TeteTransformer(nn.Module):
    """"Statement" comparison head: reads the sequence [z_a] [?] [z_b] and answers <, = or > at the "?" token.
    Number token = z x learned vector + "number" type marker; "?" token = learned vector;
    plus a side marker (left / right of the "?") and a position marker. 2 layers, 4 heads, dimension 32."""
    def __init__(self, dim_z=1, d=32, couches=2, tetes=4, n_sorties=3, sans_norme=False):
        super().__init__()
        self.val = nn.Linear(dim_z, d, bias=False)           # z x learned vector
        self.type_nombre = nn.Parameter(torch.randn(d) * 0.02)
        self.question = nn.Parameter(torch.randn(d) * 0.02)
        self.cote = nn.Parameter(torch.randn(2, d) * 0.02)    # left, right
        self.pos = nn.Parameter(torch.randn(3, d) * 0.02)
        c = nn.TransformerEncoderLayer(d, tetes, dim_feedforward=2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(c, couches, enable_nested_tensor=False)
        self.sortie = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_sorties))
        if sans_norme:  # variant: no layer normalization (the magnitude of z is not squashed)
            self.tr = nn.Sequential(*[CoucheSansNorme(d, tetes) for _ in range(couches)])
            self.sortie[0] = nn.Identity()

    def forward(self, z1, z2):
        a = self.val(z1) + self.type_nombre + self.cote[0]
        b = self.val(z2) + self.type_nombre + self.cote[1]
        q = self.question.expand(len(z1), -1)
        x = torch.stack([a, q, b], 1) + self.pos
        return self.sortie(self.tr(x)[:, 1])


class ComparateurTransformer(nn.Module):
    def __init__(self, largeur=8, dim_z=1, n_sorties=3, sans_norme=False):
        super().__init__()
        from reduction import Encodeur
        self.n_sorties = n_sorties
        self.enc = Encodeur(largeur, dim_z)
        self.tete = TeteTransformer(dim_z, n_sorties=n_sorties, sans_norme=sans_norme)

    def comparer(self, z1, z2):
        l = self.tete(z1, z2)
        if self.n_sorties == 2:
            l = torch.cat([l[:, :1], torch.full_like(l[:, :1], -1e4), l[:, 1:]], 1)
        return l


@torch.no_grad()
def mesures_au_dela(m, N, Xg, yg, modeg, rng, n_paires=200, n_min=1):
    m.eval()
    z = encoder(m, Xg)
    zn = z.cpu().numpy()
    dans = (yg <= N) & (yg >= n_min)
    sc = StandardScaler().fit(zn[dans])
    pr = LinearRegression().fit(sc.transform(zn[dans]), yg[dans]).predict(sc.transform(zn))
    out = {}
    for k, nom, hi in [(1, "taille", 20), (0, "aire", 30)]:
        idx = {n: np.where((yg == n) & (modeg == k))[0] for n in range(n_min, hi + 1)}
        lect = {n: float(pr[idx[n]].mean()) for n in range(n_min, hi + 1)}
        voisins, egal_nn = {}, {}
        for n in range(1, hi + 1):
            i = rng.choice(idx[n], n_paires); j = rng.choice(idx[n], n_paires)
            egal_nn[n] = float((m.comparer(z[i], z[j]).argmax(1) == EGAL).float().mean())
            if n < hi:
                j = rng.choice(idx[n + 1], n_paires)
                l1, l2 = m.comparer(z[i], z[j]).float(), m.comparer(z[j], z[i]).float()
                voisins[n] = {
                    "bon_cote": ((l1[:, DROITE] > l1[:, GAUCHE]).float().mean()
                                 + (l2[:, GAUCHE] > l2[:, DROITE]).float().mean()).item() / 2,
                    "egal": ((l1.argmax(1) == EGAL).float().mean() + (l2.argmax(1) == EGAL).float().mean()).item() / 2}
        bandes = {}
        for nb, (lo, hb) in {"dans_1_N": (1, N), "N+1_2N": (N + 1, min(2 * N, hi)), "au_dela_2N": (2 * N + 1, hi)}.items():
            if lo >= hb:
                continue
            a = rng.integers(lo, hb + 1, 6000); b = rng.integers(lo, hb + 1, 6000)
            g = a != b; a, b = a[g], b[g]
            i = np.array([rng.choice(idx[x]) for x in a]); j = np.array([rng.choice(idx[x]) for x in b])
            l = m.comparer(z[i], z[j]).float()
            bandes[nb] = {"bon_cote": float(((l[:, GAUCHE] > l[:, DROITE]).cpu().numpy() == (a > b)).mean()),
                          "exact": float((l.argmax(1).cpu().numpy() == cible(a, b)).mean())}
        # symmetry: is the answer to [b] ? [a] the mirror of the answer to [a] ? [b]?
        miroir = torch.tensor([DROITE, EGAL, GAUCHE], device=z.device)
        sym = {}
        for ns, (lo, hb) in {"dans_plage": (max(n_min, 1), N), "au_dela": (N + 1, hi)}.items():
            a = rng.integers(lo, hb + 1, 3000); b = rng.integers(lo, hb + 1, 3000)
            i = np.array([rng.choice(idx[x]) for x in a]); j = np.array([rng.choice(idx[x]) for x in b])
            p1 = m.comparer(z[i], z[j]).argmax(1); p2 = m.comparer(z[j], z[i]).argmax(1)
            sym[ns] = float((miroir[p1] == p2).float().mean())
        portee = N
        for n in range(N + 1, hi + 1):
            if abs(lect[n] - n) <= 0.5 and voisins[n - 1]["bon_cote"] > 0.95:
                portee = n
            else:
                break
        out[nom] = {"lecture": lect, "voisins": voisins, "egal_n_n": egal_nn, "bandes": bandes,
                    "portee": portee, "facteur": portee / N, "plafond_mesure": hi, "symetrie": sym}
    m.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_max", type=int, required=True)
    ap.add_argument("--largeur", type=int, default=8)
    ap.add_argument("--dim_z", type=int, default=1)
    ap.add_argument("--tete", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sans_egal", action="store_true")
    ap.add_argument("--bs", type=int, default=384)
    ap.add_argument("--zero", action="store_true", help="add 0 (blank image) to the range")
    ap.add_argument("--tete_type", choices=["mlp", "transformer", "transformer_sans_norme"], default="mlp")
    args = ap.parse_args()
    N = args.n_max
    n0 = 0 if args.zero else 1
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    nom = f"N{N}_L{args.largeur}_z{args.dim_z}_t{args.tete}_{'se' if args.sans_egal else 'eg'}_seed{args.seed}"
    if args.zero:
        nom = nom.replace(f"N{N}_", f"N0-{N}_")
    if args.tete_type.startswith("transformer"):
        nom = nom.replace(f"_t{args.tete}_", "_ttf_" if args.tete_type == "transformer" else "_ttfsn_")
    out = f"results/plage/{nom}"
    os.makedirs(out, exist_ok=True)

    X, y, _ = charger("train_plage15", dev)
    par_classe = int((y == 1).sum())
    X = X[: N * par_classe]
    if args.zero:  # class 0 first: indices (n - 0) * par_classe
        X = torch.cat([torch.zeros_like(X[:par_classe]), X])
    inegales = np.array([(a, b) for a in range(n0, N + 1) for b in range(n0, N + 1) if a != b])
    egales = np.array([(n, n) for n in range(n0, N + 1)])
    n_eg = 0 if args.sans_egal else args.bs // 3

    Xv, yv, _ = charger("test_plage15", dev)
    garde = (yv <= N)
    Xv, yvn = Xv[garde], yv[garde].cpu().numpy()
    if args.zero:
        Xv = torch.cat([torch.zeros_like(Xv[:1000]), Xv]); yvn = np.concatenate([np.zeros(1000, int), yvn])
    rv = np.random.default_rng(500 + args.seed)
    nv = 6000
    pv = [inegales[rv.integers(len(inegales), size=nv - (0 if args.sans_egal else nv // 3))]]
    if not args.sans_egal:
        pv.insert(0, egales[rv.integers(len(egales), size=nv // 3)])
    pv = np.concatenate(pv)
    idxv = {n: np.where(yvn == n)[0] for n in range(n0, N + 1)}
    iv = torch.tensor([rv.choice(idxv[a]) for a in pv[:, 0]], device=dev)
    jv = torch.tensor([rv.choice(idxv[b]) for b in pv[:, 1]], device=dev)
    tv = torch.tensor(cible(pv[:, 0], pv[:, 1]), device=dev)

    g = np.load("data/grand_mix.npz")
    Xg = torch.tensor(g["X"], dtype=torch.uint8, device=dev).unsqueeze(1)
    yg, modeg = g["y"], g["mode"]
    if args.zero:  # blank images (100 per mode) to fit the line on 0..N
        Xg = torch.cat([torch.zeros_like(Xg[:200]), Xg])
        yg = np.concatenate([np.zeros(200, int), yg]); modeg = np.concatenate([np.repeat([0, 1], 100), modeg])
    rg = np.random.default_rng(700 + args.seed)

    if args.tete_type.startswith("transformer"):
        m = ComparateurTransformer(args.largeur, args.dim_z, 2 if args.sans_egal else 3,
                                   sans_norme=args.tete_type == "transformer_sans_norme").to(dev)
    else:
        m = Comparateur(args.largeur, args.dim_z, 2 if args.sans_egal else 3, args.tete).to(dev)
    n_enc = sum(p.numel() for p in m.enc.parameters()); n_tete = sum(p.numel() for p in m.tete.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / MONTEE))
    print(f"{nom} : encodeur {n_enc} paramètres, tête {n_tete}", flush=True)

    suivi, gen = [], {}
    meilleur, meilleur_pas, etat, sans_progres, pas_100 = float("inf"), 0, None, 0, None
    t0 = time.time()
    for step in range(1, PLAFOND + 1):
        m.train()
        tirage = [inegales[np.random.randint(len(inegales), size=args.bs - n_eg)]]
        if n_eg:
            tirage.insert(0, egales[np.random.randint(len(egales), size=n_eg)])
        ab = np.concatenate(tirage)
        a = torch.tensor(ab[:, 0], device=dev); b = torch.tensor(ab[:, 1], device=dev)
        ia = (a - n0) * par_classe + torch.randint(par_classe, (args.bs,), device=dev)
        ib = (b - n0) * par_classe + torch.randint(par_classe, (args.bs,), device=dev)
        t = torch.tensor(cible(ab[:, 0], ab[:, 1]), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = m.enc(augmenter(flottant(torch.cat([X[ia], X[ib]]))))
            logits = m.comparer(z[:args.bs], z[args.bs:])
        loss = F.cross_entropy(logits.float(), t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()

        if step in PAS_GENERALISATION:
            gen[step] = mesures_au_dela(m, N, Xg, yg, modeg, rg, n_min=n0)
        if step % EVAL_TOUS == 0:
            m.eval()
            with torch.no_grad():
                zv = encoder(m, Xv)
                l = m.comparer(zv[iv], zv[jv]).float()
                pv_loss = F.cross_entropy(l, tv).item(); pv_acc = (l.argmax(1) == tv).float().mean().item()
            suivi.append({"pas": step, "perte_val": pv_loss, "acc_val": pv_acc, "s": time.time() - t0})
            if pv_acc == 1.0 and pas_100 is None:
                pas_100 = step
            seuil = 0.01 * meilleur if meilleur >= 1e-3 else 5e-4
            if meilleur - pv_loss > seuil or meilleur == float("inf"):
                meilleur, meilleur_pas, sans_progres = pv_loss, step, 0
                etat = copy.deepcopy(m.state_dict())
            else:
                sans_progres += 1
            if step % 500 == 0:
                print(f"pas {step:5d}  perte val {pv_loss:.4f}  exactitude val {pv_acc:.4f}  ({time.time() - t0:.0f} s)",
                      flush=True)
            if sans_progres >= PATIENCE:
                break
    pas_arret, duree = step, time.time() - t0
    m.load_state_dict(etat)
    torch.save(m.state_dict(), f"{out}/modele.pt")

    # Within 1..N: comparison on all pairs (test images), z -> n probe
    m.eval()
    with torch.no_grad():
        zt = encoder(m, Xv)
    rr = np.random.default_rng(900 + args.seed)
    paires = {}
    for a_ in range(n0, N + 1):
        for b_ in range(n0, N + 1):
            i = rr.choice(idxv[a_], 500); j = rr.choice(idxv[b_], 500)
            p = m.comparer(zt[i], zt[j]).argmax(1).cpu().numpy()
            paires[f"{a_}-{b_}"] = {"acc": float((p == cible(a_, b_)).mean()),
                                    "repart": [float((p == k).mean()) for k in range(3)]}
    ztn = zt.cpu().numpy()
    perm = rr.permutation(len(yvn)); tr, te = perm[: len(perm) // 2], perm[len(perm) // 2:]
    sc = StandardScaler().fit(ztn[tr])
    prd = LinearRegression().fit(sc.transform(ztn[tr]), yvn[tr]).predict(sc.transform(ztn[te]))
    r2 = float(1 - ((prd - yvn[te]) ** 2).sum() / ((yvn[te] - yvn[te].mean()) ** 2).sum())
    ineg = float(np.mean([v["acc"] for k, v in paires.items() if k.split("-")[0] != k.split("-")[1]]))
    egal = float(np.mean([v["acc"] for k, v in paires.items() if k.split("-")[0] == k.split("-")[1]]))
    gen["arret"] = mesures_au_dela(m, N, Xg, yg, modeg, rg, n_min=n0)

    metrics = {"nom": nom, "n_max": N, "largeur": args.largeur, "dim_z": args.dim_z, "tete": args.tete, "tete_type": args.tete_type,
               "sans_egal": args.sans_egal, "zero": args.zero, "seed": args.seed, "params_encodeur": n_enc, "params_tete": n_tete,
               "pas_arret": pas_arret, "meilleur_pas": meilleur_pas, "pas_100_val": pas_100, "duree_s": duree,
               "suivi": suivi, "dans_1_N": {"inegales": ineg, "egales": egal, "sonde_r2": r2, "paires": paires},
               "au_dela": {str(k): v for k, v in gen.items()}}
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=1)
    gt, ga = gen["arret"]["taille"], gen["arret"]["aire"]
    bt = gt["bandes"]
    print(f"RÉSULTAT {nom} | {n0}-{N} inégales {ineg:.3f} égales {egal:.3f} R² {r2:.3f} | taille : portée {gt['portee']} "
          f"(×{gt['facteur']:.2f}), N+1..2N bon côté {bt.get('N+1_2N', {}).get('bon_cote', float('nan')):.3f}, "
          f">2N {bt.get('au_dela_2N', {}).get('bon_cote', float('nan')):.3f}, lit 20 → {gt['lecture'][20]:.1f} | "
          f"aire : portée {ga['portee']}, lit 30 → {ga['lecture'][30]:.1f} | symétrie {gt['symetrie']['dans_plage']:.3f}/{gt['symetrie']['au_dela']:.3f} | arrêt {pas_arret}", flush=True)


if __name__ == "__main__":
    main()
