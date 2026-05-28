# REST: Representative-batch Ensemble Self-Training for Semi-Supervised Medical Image Classification

Official code for the paper **"REST: Representative-batch Ensemble Self-Training
Framework for Semi-Supervised Medical Image Classification."**

> **Note:** Link to the paper / arXiv will be added here once available.

---

## Overview

REST is a semi-supervised self-training framework for medical image
classification with very few labels. It extends Sequential Retraining and
Pseudo-labeling in Mini-batches (SRPM-ST) with three modifications:

1. **Augmentation-based ensemble.** The single base classifier is replaced by
   three jointly trained ResNet-18 networks. Each network sees a complementary
   augmented view of the same image (clean, noise-based, and geometric). The
   three networks are trained together with a shared optimizer by averaging
   their logits, and their predictions are combined by soft voting at inference.
2. **Representative mini-batch selection (TypiClust).** Instead of random
   mini-batches, REST extracts penultimate-layer features from the current
   ensemble and runs *k*-means on the unlabeled pool, selecting the sample
   nearest each cluster centroid.
3. **Class-weighted loss with pseudo-label attenuation.** A class-weighted
   cross-entropy loss down-weights pseudo-labeled samples (by a factor
   `alpha`) to mitigate class imbalance and label noise.

REST is evaluated on three medical image datasets — **BreastMNIST**,
**ISIC Skin Cancer**, and **ACRIMA (glaucoma)** — using only 1–10% of the
labels, and consistently outperforms the initial classifier, single/ensemble
full-batch self-training (F-ST), and the consistency-based SSL methods
FixMatch and FlexMatch.

---

## Repository structure

```
REST/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   └── hyperparameters.md        # per-dataset hyperparameters (Table 1 of the paper)
└── src/
    ├── breastmnist/
    │   ├── run_srpm.py           # multi-seed REST runner (main experiment)
    │   └── tune.py               # mini-batch count (J) tuning
    ├── skincancer/
    │   ├── run_srpm.py
    │   └── tune.py
    └── acrima/
        ├── run_srpm.py
        └── tune.py
```

Each dataset has its own self-contained pair of scripts: `tune.py` selects the
number of mini-batches `J` on the validation set, and `run_srpm.py` runs the
final multi-seed REST experiment.

---

## Installation

```bash
git clone <REPO_URL>
cd REST
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Experiments were run with Python 3.10 and PyTorch on NVIDIA H100 GPUs.

---

## Datasets

| Dataset      | Task                          | Source |
|--------------|-------------------------------|--------|
| BreastMNIST  | Binary breast ultrasound      | Downloaded automatically via the `medmnist` package |
| ISIC Skin Cancer | Binary benign vs. malignant | ISIC Archive — https://www.isic-archive.com/ |
| ACRIMA       | Binary glaucoma (fundus)      | https://figshare.com/articles/dataset/CNNs_for_Automatic_Glaucoma_Assessment_using_Fundus_Images_An_Extensive_Validation/7613135 |

**BreastMNIST** requires no manual download; `medmnist` fetches it on first run.

**Skin Cancer** and **ACRIMA** expect an `ImageFolder`-style layout:

```
<data_root>/
├── train/
│   ├── class_0/   *.jpg|*.png
│   └── class_1/
└── test/
    ├── class_0/
    └── class_1/
```

Set the dataset location before running, either by editing the `DATA_ROOT`
variable near the top of the corresponding `run_srpm.py` / `tune.py`, or by
exporting an environment variable if you adapt the scripts to read one:

```bash
export REST_DATA_ROOT=/path/to/skin     # or /path/to/glaucoma
```

---

## Reproducing the results

### 1. (Optional) Tune the number of mini-batches `J`

```bash
python src/breastmnist/tune.py
python src/skincancer/tune.py
python src/acrima/tune.py
```

Each `tune.py` sweeps a dataset-specific candidate range for `J` (see
`configs/hyperparameters.md`) and reports the value that maximizes validation
AUC. The optimal per-seed values we used are already filled into the
`SEEDS`/`NUM_BATCHES` lists in each `run_srpm.py`.

### 2. Run the main REST experiment

```bash
python src/breastmnist/run_srpm.py
python src/skincancer/run_srpm.py
python src/acrima/run_srpm.py
```

Each runner trains REST across three random seeds (42, 43, 44) and writes
per-seed JSON/`.txt` summaries and an aggregated mean ± std summary to a
timestamped folder under `results_*`.

---

## Key hyperparameters

These are fixed across the main experiments (full details in
`configs/hyperparameters.md` and Table 1 of the paper):

- Pseudo-label confidence threshold: `tau = 0.75`
- Pseudo-label loss weight: `alpha = 0.5`
- Optimizer: AdamW, weight decay `1e-1`, cosine annealing schedule
- Max epochs: 30, early stopping patience: 7 (on validation AUC)
- Backbone: modified ResNet-18 with channels `[22, 44, 88, 176]` (~1.32–1.4M params per model)
- Evaluation metric: Area Under the ROC Curve (AUC), mean ± std over 3 seeds

---

## Citation

If you use this code, please cite:

```bibtex
@article{moldagazyyev_rest,
  title   = {REST: Representative-batch Ensemble Self-Training Framework for
             Semi-Supervised Medical Image Classification},
  author  = {Moldagazyyev, Ruslan and Jamwal, Prashant K. and Mukhamediya, Azamat},
  journal = {TBD},
  year    = {2026}
}
```

This work builds on SRPM-ST:

```bibtex
@article{mukhamediya2024srpmst,
  title   = {SRPM-ST: Sequential retraining and pseudo-labeling in mini-batches
             for self-training},
  author  = {Mukhamediya, Azamat and Zollanvari, Amin},
  journal = {Neurocomputing},
  year    = {2024}
}
```

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
