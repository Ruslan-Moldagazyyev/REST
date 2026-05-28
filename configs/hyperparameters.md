# Dataset-specific hyperparameters

This mirrors Table 1 of the paper. Values are already set in the corresponding
`run_srpm.py` / `tune.py` scripts.

| Parameter                  | BreastMNIST            | Skin Cancer            | ACRIMA (Glaucoma)      |
|----------------------------|------------------------|------------------------|------------------------|
| Input channels `C`         | 1 (grayscale)          | 3 (RGB)                | 3 (RGB)                |
| Image size `H x W`         | 28 x 28                | 64 x 64                | 64 x 64                |
| Labeled fraction `rho`     | 0.10                   | 0.01                   | 0.02                   |
| Pseudo-label weight `alpha`| 0.5                    | 0.5                    | 0.5                    |
| Confidence threshold `tau` | 0.75                   | 0.75                   | 0.75                   |
| Learning rate `eta`        | 1e-3                   | 1e-4                   | 1e-4                   |
| Weight decay               | 1e-1                   | 1e-1                   | 1e-1                   |
| Max epochs                 | 30                     | 30                     | 30                     |
| Early stopping patience    | 7                      | 7                      | 7                      |
| Training batch size        | 16                     | 16                     | 16                     |
| Evaluation batch size      | 32                     | 32                     | 32                     |
| `J` candidate range        | {10, 12, ..., 32}      | {28, 30, ..., 40}      | {28, 30, ..., 40}      |
| Geometric rotation         | +/- 15 deg             | +/- 180 deg            | +/- 180 deg            |
| Ensemble params (total)    | ~4.0M                  | ~4.2M                  | ~4.2M                  |
| Per-model params           | ~1.32M                 | ~1.4M                  | ~1.4M                  |

All experiments use seeds **42, 43, 44**.
