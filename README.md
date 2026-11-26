# 🛡️ FraudGuard ML | Real-Time Bank Fraud Detection
 
> End-to-end platform: calibrated XGBoost model, FastAPI scoring API, 3-band decision (Authorized / Review / Blocked) and an analytics dashboard with explanations for fraud analysts.
 
![Dashboard](docs/screenshots/dashboard.png)
 
---
 
## ✨ Key Features
 
- **Complete pipeline**: training → versioned artifact → scoring API → real-time dashboard
- **Chronological validation**: train 70% / calibration 7.5% / thresholds 7.5% / test 15%, with no temporal mixing
- **Platt calibration on the XGBoost margin**: true probabilities, usable for setting business thresholds
- **Business-cost-driven thresholds** (false-negative cost depending on the amount), then translated into 3 decision bands
- **Causal features at inference**: a card's history only contains strictly earlier transactions
- **Explainability**: natural-language analysis (warning signals / reassuring signals) and behavioral metrics per transaction
- **Automatic training reports**: feature quality (PSI, univariate AUC), stability by period, missed frauds
---
 
## 📊 Results
 
Trained and evaluated on the public **Credit Card Transactions Fraud Detection** dataset (Kaggle, `fraudTrain`, simulated Sparkov-type bank transactions, 1,296,675 transactions, 0.58% fraud).
Evaluated on the **final chronological test set: 194,502 transactions, 1,133 frauds** (never used for calibration or threshold selection).
 
| Metric | Value |
|---|---|
| PR-AUC | **0.981** |
| ROC-AUC | **0.9997** |
| Recall (cost threshold 0.02) | **98.2%** |
| Precision (cost threshold 0.02) | 69.2% |
| F1-score | 0.812 |
| Fraudulent amount detected | $601,623 |
 
### 3-band decision (test)
 
| Band | Share of transactions | Frauds | Role |
|---|---|---|---|
| ✅ Authorized (score < 20) | 98.94% | 14 | Automatic approval |
| 🟡 Manual review (20–80) | 0.48% (≈ 4.75 / 1000 tx) | 49 | Analyst |
| 🔴 Blocked (score ≥ 80) | 0.58% | 1,070 | Automatic block |
 
- **98.76%** of frauds are intercepted (Blocked + Review)
- **94.4%** of frauds are blocked automatically, with a blocking precision of **94.4%**
- Review workload: **0.48%** of the flow, well below the 5% cap
### Stability over time (4 chronological slices of the test set)
 
| Period | Transactions | Frauds | PR-AUC | Recall (Blocked + Review) | Blocking precision |
|---|---|---|---|---|---|
| 1 | 48,626 | 209 | 0.969 | 98.6% | 93.7% |
| 2 | 48,626 | 351 | 0.995 | 100% | 95.0% |
| 3 | 48,625 | 261 | 0.982 | 98.9% | 92.5% |
| 4 | 48,625 | 312 | 0.972 | 97.4% | 95.7% |
 
### Evaluation curves
 
| ROC | Precision–Recall |
|---|---|
| ![ROC](reports/roc_curve.png) | ![PR](reports/precision_recall_curve.png) |
 
| Confusion matrix | Feature importance |
|---|---|
| ![CM](reports/confusion_matrix.png) | ![FI](reports/feature_importance_selected.png) |
 
---
 
## 🧠 ML Approach
 
### Features (15 selected out of 19 candidates)
 
| Family | Variables |
|---|---|
| Amount | `amt`, `amt_vs_category_z`, `amt_zscore_card`, `amt_to_card_mean`, `amt_to_card_max` |
| Velocity | `nb_tx_24h`, `sum_amt_24h`, `sum_amt_1h`, `tx_accel_1h_vs_24h`, `amt_share_of_24h`, `time_since_last_tx` |
| Behavior | `is_new_category`, `category_code`, `hour`, `hour_share_card` |
 
**Two-step selection**: a "scout" model ranks features by cumulative gain (95% threshold), then the card's personal-behavior features are forced into the final model.
**Safeguards**: features with potential leakage (`merchant_freq`, `category_share_card`) and those whose value would be unreliable at inference (speed, distance to the previous merchant) are deliberately excluded.
 
### Modeling
 
- **XGBoost** (400 trees, depth 6) with a softened `scale_pos_weight` (`(n_neg/n_pos)^0.5`) to handle class imbalance
- **Platt calibration on the raw margin** rather than on `predict_proba`: avoids probabilities being crushed to the floor and keeps resolution at the low end of the scale
- **0–100 risk score on a logarithmic scale** to finely distinguish very low probabilities
- **Isolation Forest** trained and evaluated as an unsupervised signal (ROC-AUC 0.967)
### Business-driven thresholds
 
| Threshold | Rule |
|---|---|
| Decision threshold | Best F1 among thresholds whose estimated cost is within 5% of the minimum (FN: $200 + 0.5 × amount, bounded to [50, 2000]; FP: $10) |
| `t_low` (Authorized → Review) | 99.5th percentile of legitimate-transaction probabilities (validation) |
| `t_high` (Review → Blocked) | Smallest threshold guaranteeing precision ≥ 95% (validation) |
| Review cap | 5% of the flow at most |
 
---
 
## ⚙️ Scoring API
 
- **FastAPI** + **SQLite**: every transaction is scored, logged and added to the card's memory
- **Causal features**: 1 h / 6 h / 24 h / 7 d sliding windows computed on history strictly prior to the transaction
- **Training / inference parity**: time units, standard deviation (ddof = 1) and per-category statistics are learned at training time and reloaded from the artifact
- **Artifact checked** at load time (keys, consistency `p_floor < t_low < t_high`)
- **Initial memory**: replay of a transaction history before the live stream (`historique.py`)
## 🖥️ Dashboard
 
- **Real-time view** of scored transactions, with search by ID, merchant, card or city
- **Time analysis**: decision breakdown, risk-score distribution, 7-day trend
- **Geographic analysis**: world map with volume and fraud rate per city, filters by continent
- **Transaction sheet**: score, decision, AI analysis, behavioral metrics (z-score, velocity, merchant affinity, inter-transaction interval)
- **Per-card history**, JSON export, authentication (passwords hashed with PBKDF2-SHA256, 200,000 iterations)
---
 
## 🏗️ Architecture
 
```
 Kaggle dataset ──► Train_ML.py ──► models/fraud_xgb_model.joblib + reports/
                                         │
 Ingestion.py ──► API_Score.py (FastAPI :8001) ──► SQLite (fraud.db)
 (simulated stream) causal features + scoring               │
                  3-band decision                           ▼
                                              dashboard.py (FastAPI :8002)
```
 
**Stack**: Python · XGBoost · scikit-learn · pandas · FastAPI · SQLite · Uvicorn · HTML/CSS/JS
 
---
 
## 🔁 Demo Stream
 
To feed the dashboard, an **ingestion simulator** (`Ingestion.py`) continuously generates a stream of transactions to the scoring API:
 
- **Customers**: one fixed home location per card, read from `data/sample/History.csv`
- **Transaction locations**: 79 cities around the world
- **Behaviors**: normal transactions, benign anomalies and labeled fraud scenarios: `IMPOSSIBLE_TRAVEL`, `EXTREME_AMOUNT`, `CARD_TESTING`, `VELOCITY_ATTACK`, `NIGHT_HIGH_RISK`
It lets you watch the system react live, without any real banking data. The `simulate_world.py` script also generates a labeled world dataset (ground truth included) to evaluate the model on other card profiles.
 
---
 
## 🚀 Installation
 
```bash
git clone https://github.com/Developer-Dynamo/transaction-fraud-detection.git
cd transaction-fraud-detection
 
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
 
1. **Data**: download the Kaggle dataset and place the prepared file in `data/` (see `data/README.md`)
2. **Training**: `python src/Train_ML.py --data data/fraudTrain_clean.csv` (generates `models/` and `reports/`)
3. **Scoring API**: `python src/API_Score.py` (port 8001)
4. **History**: `python src/historique.py --csv data/sample/History.csv`
5. **Live stream**: `python src/Ingestion.py`
6. **Dashboard**: `python src/dashboard.py` (port 8002), then create an account at `/login`
The same commands are available through `make` (`make train`, `make api`, `make seed`, `make stream`, `make dashboard`).
Settings are configured via environment variables (see `.env.example`).
 
### With Docker
 
```bash
docker compose up --build                      # API + dashboard
docker compose --profile stream up ingestion   # simulated transaction stream
```
 
---
 
## 📁 Project Structure
 
```
├── src/
│   ├── Train_ML.py          # Training, calibration, thresholds, reports
│   ├── calibration.py       # Platt calibrator on the XGBoost margin
│   ├── features_builder.py  # Feature computation (causal)
│   ├── API_Score.py         # Scoring API (FastAPI)
│   ├── dashboard.py         # Web dashboard
│   ├── Ingestion.py         # Transaction stream simulator
│   ├── historique.py        # History replay before the stream
│   ├── simulate_world.py    # Labeled simulated world dataset
│   └── world_cities.py      # Shared cities
├── data/                    # README + sample (data/sample/History.csv)
├── models/                  # Model artifact
├── reports/                 # Metrics, charts, quality reports
├── docs/screenshots/        # Dashboard screenshots
├── Dockerfile · docker-compose.yml · Makefile · requirements.txt
└── .github/workflows/ci.yml
```
 
---
 
## 🗺️ Roadmap
 
- [ ] Reintroduce speed and distance to the previous merchant (impossible travel) once training / inference parity is validated
- [ ] Per-transaction SHAP explainability
- [ ] Benchmark on a second public dataset
- [ ] Comparison baselines (logistic regression, business rules)
- [ ] Drift monitoring (PSI) and automatic retraining
---
 
## 👤 Author
 
**Oussama Hajji** — [GitHub](https://github.com/Developer-Dynamo) — [LinkedIn](https://linkedin.com/in/oussama-hajji-ensi)
 
