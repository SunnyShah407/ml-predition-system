# Regression System (Tender E-Commerce Use Case)

This project benchmarks multiple **regression algorithms** using a procurement-focused use case.

## Use case
**A Tender E-Commerce System** is a digital procurement platform that streamlines the end-to-end tendering process between an **Organization (Buyer)** and **Suppliers (Vendors)**. It automates tender creation, evaluation, competitive price bidding, and contract finalization — replacing manual, paper-based procurement with a transparent, auditable digital workflow.

## Algorithms covered (Regression)
- Linear Regression (baseline)
- Ridge Regression
- Lasso Regression
- SVR (Support Vector Regression)
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

## Project structure
```
.
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── figures/
│   └── metrics/
├── scripts/
└── src/
    └── mlps/
        ├── data/
        ├── features/
        └── models/
```

## Quickstart

Create env + install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate a realistic synthetic tender dataset:

```bash
python scripts/generate_synthetic_tender_data.py --config configs/synthetic_tender.yaml
```

Train + evaluate all models:

```bash
python scripts/train_regression.py --config configs/regression_tender.yaml
```

Enable XGBoost (optional):
- `xgboost` may require OpenMP on macOS. If import fails, install it with `brew install libomp`, then set `models.xgboost: true` in `configs/regression_tender.yaml`.

Outputs:
- `reports/metrics/metrics.json` (per-model metrics)
- `reports/figures/` (diagnostic plots)
- `artifacts/` (trained models + preprocessors)

