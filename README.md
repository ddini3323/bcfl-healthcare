# BCFL Healthcare: Blockchain-Enabled Federated Learning for Chest X-ray Diagnosis

Implements all four project objectives on the NIH ChestX-ray14 dataset:

1. **Federated Learning model** for a simulated multi-hospital healthcare environment (`federated/`, `models/`)
2. **Blockchain-Enabled FL (BCFL)**: a real hash-chained ledger logging every client update, with tamper detection (`blockchain/`)
3. **Hybrid privacy-preserving mechanisms**: Differential Privacy (clipping + calibrated noise) combined with Secure Aggregation (pairwise cancelling masks) (`privacy/`)
4. **Testing & validation** on the NIH ChestX-ray14 dataset with per-class AUC-ROC/F1 and a centralized-vs-FL-vs-BCFL comparison (`evaluation/`)

## Project layout

```
bcfl-healthcare/
├── config.py              # all paths & hyperparameters -- edit this first
├── main.py                 # run one full BCFL training + evaluation
├── run_comparison.py       # run centralized vs FL-only vs BCFL+privacy, side by side
├── test_pipeline.py        # smoke test on synthetic data -- run this first
├── requirements.txt
├── data/
│   └── dataset.py          # NIH CSV loading, CLAHE preprocessing, Keras Sequence generator
├── models/
│   └── model.py             # DenseNet121 transfer-learning model + lightweight CNN
├── federated/
│   ├── client.py            # local training: plain FedAvg or FedProx
│   └── server.py            # weighted aggregation
├── privacy/
│   └── mechanisms.py        # DP clip+noise, secure-aggregation masks, epsilon estimate
├── blockchain/
│   └── ledger.py             # hash-chained block ledger + tamper detection
├── evaluation/
│   └── evaluate.py           # AUC/F1 per class, experiment comparison
├── checkpoints/              # saved models land here
└── outputs/                  # metrics JSON, blockchain ledger JSON, comparison chart land here
```

## Setup (VS Code / local machine)

1. **Python**: 3.10 or 3.11 recommended (TensorFlow 2.15/2.16 wheels are readily available for these).

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Download the dataset** — two options:

   **Option A (recommended): kagglehub, handles auth + caching for you**
   ```bash
   pip install kagglehub   # already in requirements.txt
   python download_dataset.py
   ```
   First run will prompt for Kaggle credentials if you don't have `~/.kaggle/kaggle.json` set up yet (get one free at https://www.kaggle.com/settings -> "Create New Token"). The script downloads (~45GB, one-time, cached after that) and prints the local path.

   `config.py` will **automatically find this cached path** the next time you run anything — you don't need to set `CXR_DATA_DIR` manually if you used `download_dataset.py`. It only downloads once; subsequent runs reuse the kagglehub cache instantly.

   **Option B: manual download** from [NIH ChestX-ray14 on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data), unzip it somewhere, then point `config.py` at it:
   ```bash
   # Windows PowerShell
   $env:CXR_DATA_DIR = "C:\Users\you\Downloads\nih_chest_xrays"
   # macOS/Linux
   export CXR_DATA_DIR=/path/to/nih_chest_xrays
   ```

   Either way you need at minimum: `Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt`, and the `images_001/` ... `images_012/` folders (each with an `images/` subfolder of PNGs).

5. **Run the smoke test first** — this does not need the real dataset, just confirms your environment is set up correctly:
   ```bash
   python test_pipeline.py
   ```
   You should see all 7 steps print "OK" and a final "All pipeline components work correctly."

## Running the real pipeline

Quick test run on the lightweight CNN (fast, works fine on a laptop CPU):
```bash
python main.py --model lightweight --rounds 3 --clients 3
```

Full run with the DenseNet121 transfer-learning model and FedProx (recommended for your actual results — a GPU is strongly recommended for this):
```bash
python main.py --model densenet --rounds 10 --clients 5 --fedprox
```

Disable privacy to get a plain-FL baseline for comparison:
```bash
python main.py --model lightweight --rounds 5 --no-privacy --tag fl_baseline
```

### Three-way comparison (centralized vs FL vs BCFL+privacy)
This is the evidence for objectives 3 & 4 — it shows the accuracy cost of adding privacy/decentralization:
```bash
python run_comparison.py --model lightweight --rounds 5 --clients 3
```
Produces `outputs/comparison_results.json` and `outputs/comparison_chart.png`.

## What each objective maps to in the code

| Objective | Where |
|---|---|
| 1. FL model | `models/model.py` (architectures), `federated/client.py` (FedAvg + FedProx local training), `federated/server.py` (weighted aggregation) |
| 2. Blockchain integration | `blockchain/ledger.py` — real hash-chained blocks, `is_valid()` detects tampering |
| 3. Hybrid privacy | `privacy/mechanisms.py` — DP (clip + Gaussian noise) + Secure Aggregation (pairwise masks), combined in `apply_hybrid_privacy()` |
| 4. Test & validate | `evaluation/evaluate.py` — per-class AUC-ROC/F1 on held-out test set; `run_comparison.py` — centralized vs FL vs BCFL comparison |

## Notes on what's simplified vs. "real world"

Be upfront about these if this goes into a report/thesis defense — reviewers will ask:

- **Blockchain**: this is a single-process, in-memory hash chain (no consensus protocol, no distributed nodes). It correctly demonstrates the integrity property (tamper-evidence via chained hashes) but is not a deployed distributed ledger. State this explicitly rather than calling it "blockchain" unqualified.
- **Secure aggregation**: the pairwise-mask cancellation is mathematically correct and demonstrates the same guarantee (server can't see individual updates) but skips the real-world key-exchange handshake (e.g., Diffie-Hellman) that production systems like Google's Secure Aggregation protocol use to establish those masks between actual separate machines.
- **DP epsilon**: `estimate_epsilon()` uses a simple composition bound for a quick, reportable number. For a rigorous accountant, integrate `tensorflow-privacy`'s RDP accountant or `opacus` (if you move training to PyTorch).
- **Federated "clients"**: these are simulated by splitting one dataset — genuine cross-silo FL would run on physically separate machines exchanging only weights over a network, not raw data in one process.

## Troubleshooting

- `ModuleNotFoundError`: make sure your virtual environment is activated and `pip install -r requirements.txt` succeeded.
- Very slow training / out of memory: use `--model lightweight` and reduce `--clients`/`--rounds`, or reduce `BATCH_SIZE` in `config.py`.
- Image loading returns nothing: check that `IMAGES_DIR` in `config.py` points at the folder containing `images_001/`, `images_002/`, ... (the loader walks all subfolders looking for `.png` files).
