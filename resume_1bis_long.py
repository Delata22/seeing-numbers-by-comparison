"""Table for step 4 (long continuation of test 1bis, 9000 more steps): reading of images and sums, image = sum.

For each model: end of 1bis (step 6000 of 1bis = step 0 of the continuation), then mean of the log over the last 3000 steps
(log every 500 steps: 6500..9000, i.e. 6 points, to smooth out fluctuations), and final value (full measurement).
Usage (from the repository root): uv run python resume_1bis_long.py -> logs/resume_1bis_long.txt
"""
import glob, json
import numpy as np

L = []
for f in sorted(glob.glob("results/au_dela_20_1bis_long/D*_general_seed*.json")):
    r = json.load(open(f)); j = r["journal"]
    fin_pts = [j[k] for k in j if int(k) >= 6500]
    L.append(f"{r['nom']}")
    for n in ("25", "28", "30"):
        moy = lambda c: np.mean([x[n][c] for x in fin_pts])
        L.append(f"    n={n} | début : image {j['0'][n]['lecture_image']:.1f}, somme {j['0'][n]['lecture_somme']:.1f}, = {j['0'][n]['image_egal_somme']:.2f}"
                 f" | 3000 derniers pas (moy. de {len(fin_pts)}) : image {moy('lecture_image'):.2f}, somme {moy('lecture_somme'):.2f}, = {moy('image_egal_somme'):.2f}"
                 f" [{min(x[n]['image_egal_somme'] for x in fin_pts):.2f}-{max(x[n]['image_egal_somme'] for x in fin_pts):.2f}]")
    a = r["apres"]
    L.append("    final (mesure complète) image = somme 21/25/28/30 : " + " / ".join(f"{a['par_n'][str(n)]['img_somme_egal']:.2f}" for n in (21, 25, 28, 30))
             + f" | dans la plage 20 : {a['dans_plage']['20']['img_somme_egal']:.2f} | somme/somme 61-100 = {a['oubli']['61-100']['egal']:.2f}")
    traj = " ".join(f"{int(k)//1000 if int(k) % 1000 == 0 else int(k)/1000:g}k:{j[k]['30']['lecture_image']:.1f}" for k in sorted(j, key=int) if int(k) % 1500 == 0)
    L.append(f"    trajectoire de la lecture de l'image de 30 (tous les 1500 pas) : {traj}")
txt = "\n".join(L); print(txt); open("logs/resume_1bis_long.txt", "w").write(txt + "\n")
