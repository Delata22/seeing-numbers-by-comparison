# Reproducing the experiments

All commands run from the repository root. Times were measured on one consumer GPU (12 GB, CUDA) with
2 to 6 runs in parallel, so a run alone is usually faster. Every script uses `cuda`.

```bash
uv sync
```

The shipped `logs/` summaries and `results/**/*.json` files are the outputs of the reported runs. Model
weights are not in git: see "Trained models" below.

## 1. Data (not in git, about 1.2 GB, regenerable)

```bash
mkdir -p data
uv run python generate.py              # data/{train,test}_{mix,uni}.npz, 64x64, 1-9 objects (~1 min)
uv run python generate_grand.py        # data/grand_{mix,uni}.npz, test images up to 30 objects
uv run python generate_plage.py        # data/{train,test}_plage15.npz, 1-15 objects
uv run python generate_plage20.py      # data/{train,test}_plage16a20.npz, 16-20 objects (expressions)
uv run python generate_21a30.py        # data/train_21a30_aire.npz, 21-30 objects (above 20)
uv run python generate_cent.py         # data/cent_X.npy, data/cent_y.npy, 192x192, 1-100 objects (GPU)
```

Each generator takes from under a minute to a few minutes. Images are separated white shapes (disc, square,
triangle) on a black background; half of them have a fixed total area and half a fixed shape size, so that
neither area nor shape size gives the number.

## 2. Order of the experiments

Later experiments start from models trained earlier. `indexer_modeles.py` copies every trained model into
`modeles/<configuration>/seed<k>.pt` (plus a JSON with metadata and `modeles/INDEX.md`), and the addition,
expression, above-20 and analysis scripts load their starting models from there. Run it after each series:

1. `run_compare.sh`, `run_reduction.sh`, `run_plage.sh` (then `resume_plage.py`), `run_tf.sh`, then `uv run python indexer_modeles.py`
2. `analyse_poids.py`, `analyse_strategies.py`
3. `run_addition.sh`, `run_expressions.sh`, then `uv run python indexer_modeles.py`
4. `run_expressions_moins.sh`, `run_au_dela_20.sh`, then `uv run python indexer_modeles.py`

The 1-100 series (`run_cent.sh`) is independent.

## 3. Experiments

### Comparison on 1 to 9 objects

A 64-dimensional latent, trained only on more / same / fewer, with held-out pairs (3-6, 5-8, 2-7) and a held-out
equality (6 = 6). Measures: accuracy on unseen pairs, transitivity, linear probe of $n$ from $z$, white-area
probe, PCA of the latent.

- Command: `./run_compare.sh` (24 runs: mix / uni datasets, 4 conditions, 3 seeds; also runs `plots.py mix` and `plots.py uni`)
- Output: `logs/resume_{mix,uni}.txt`, `figures/{mix,uni}/`
- Time: ~15 min per run (4000 steps cap)

### Encoder size reduction

Width and latent-size sweep down to one filter per layer, with and without the "same" class.

- Command: `./run_reduction.sh jobs/reduction_jobs_a.txt`, and the same for `_b`, `_c`, `_d`; then `uv run python resume_reduction.py`
- Output: `logs/reduction/resume.txt`
- Time: 1-6 min per run, 86 runs

### Training range 1..N, with and without the equality class

How far the learned number line extends beyond the training range (fixed-size test images up to 20).

- Command: `./run_plage.sh jobs/plage_jobs.txt`, `./run_plage.sh jobs/plage_jobs_N2.txt`, `./run_plage.sh jobs/plage_jobs_zero.txt`; then `uv run python resume_plage.py`
- Output: `logs/plage/resume.txt`
- Time: 1-5 min per run, 60 runs

Run `resume_plage.py` before `run_tf.sh`: it does not filter on the head type, so once the transformer runs exist
in `results/plage/N0-*`, it averages them together with the MLP runs in the 0-3, 0-5 and 0-9 rows. The shipped
`logs/plage/resume.txt` was written before the transformer runs.

### Probes and weight analyses

Controlled probe images (one disc at any position, empty image, disc radius, touching discs), spread of $z$
against $n$, first-layer filters, and linear CKA across seeds and training ranges. CPU only, reads `modeles/plage_*`.

- Command: `uv run python analyse_poids.py` and `uv run python analyse_strategies.py`
- Output: `logs/poids/resume.txt`, `figures/poids/`, `logs/strategies/resume.txt`, `figures/strategies/`
- Time: a few minutes each

### Transformer vs MLP comparison head

The same encoder and training, with a small transformer reading the two latents instead of the MLP head (with and
without layer normalisation).

- Command: `./run_tf.sh`, then `uv run python resume_tf.py`
- Output: `logs/plage/resume_tf.txt`
- Time: 2-5 min per run, 12 runs

### 1 to 100 objects

192x192 images, shape radius drawn independently of $n$, three training distributions (uniform, $1/n$, $1/n^2$).

- Command: `uv run python generate_cent.py`, then `./run_cent.sh` (runs `resume_cent.py`)
- Output: `logs/cent/resume.txt`
- Time: ~5-10 min per run (up to 8000 steps), 26 runs

### Addition, first version (a single sum always on the left)

- Command: `./run_addition.sh`
- Output: `results/addition/*/metrics.json`
- Time: ~20 min per run, 12 runs

### Expressions with "+"

A recurrent brick folds sides of one to three terms; numbers, sums and their combinations appear on both sides of
the shared comparison head. General brick and successor ($+1$ only) brick, from starting ranges 0-3 and 0-9.

- Command: `./run_expressions.sh`
- Output: `results/expressions/*/metrics.json` (means over the 3 seeds)
- Time: 30-60 min per run, 12 runs

### Expressions with "+" and "−"

A second learned symbol for "−"; running values below zero are tested but never trained.

- Command: `./run_expressions_moins.sh`
- Output: `results/expressions_moins/*/metrics.json`
- Time: 10-90 min per run, 12 runs

### Above 20

Sums of images of at most 20 objects read on the image line up to 100 (part A); fine-tuning on images of 0 to 30
(part B, with a longer-comparison control); image of $n$ against $[a]+[b]$ with a frozen brick (test 1) and with
the brick trained jointly (test 1bis), then a long continuation of the joint fine-tuning.

- Command: `./run_au_dela_20.sh`
- Output: `logs/au_dela_20.txt`, `results/au_dela_20_B/*.json`, `logs/au_dela_20_test1.txt`, `logs/resume_1bis.txt`, `logs/resume_1bis_long.txt`
- Time: ~5 min (part A); 1-5 min per fine-tuning run (24); ~8 min per joint run (12); 7-18 min per long run (6)

The launchers skip runs whose output already exists, take the number of parallel jobs as an argument, and
read their job lists from `jobs/`. Training logs go to `logs/<series>/`.

## 4. Trained models

The weights of every reported run (all seeds) are in the release asset
`seeing-numbers-by-comparison-models.zip` (about 59 MB). Unzip it at the repository root: it creates
`modeles/<configuration>/seed<k>.pt` with a JSON per seed and `modeles/INDEX.md` (best seed per configuration).
With it, `analyse_poids.py` and `analyse_strategies.py` run directly, and the addition, expression and
above-20 scripts find their starting models. Part A and test 1 of `run_au_dela_20.sh` also read
`results/expressions/` and `results/au_dela_20_B/`, which only exist after running those series.

## 5. Notes

- Printed outputs and the `logs/` summaries are in French (the original lab notebook); identifiers are in
  French too (`egal` = "same", `plage` = range, `depart` = starting model, `brique` = brick, `graine` = seed).
- The numbers are means over 3 seeds and are preliminary.
