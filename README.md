
# Fraud Risk Intelligence Pipeline

## 1. Project Overview

A production-oriented fraud-risk intelligence pipeline
for detecting suspicious financial transactions.

The system produces:

- Calibrated fraud probability
- Approve / Review / Block decision
- Transaction-level explanations
- Cost-sensitive thresholds
- Model monitoring reports

## 2. Dataset

Total transactions: 3000

Legitimate transactions: 2636

Fraud transactions: 364

Target: fraud

0 = Legitimate

1 = Fraud

The dataset does not contain an exact transaction timestamp.
Therefore, chronological validation cannot be performed.

## 3. Environment

Python 3.11 or 3.12 recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

## 4. Project Execution

Run data audit:

```bash
python src/data_audit.py
```

Run feature engineering:

```bash
python src/feature_engineering.py
```

Run model training:

```bash
python src/model_training.py
```

Run calibration:

```bash
python src/calibration.py
```

Run cost optimization:

```bash
python src/cost_optimization.py
```

Run decision engine:

```bash
python src/decision_engine.py
```

Run explainability:

```bash
python src/explainability.py
```

Run robustness:

```bash
python src/robustness.py
```

Run monitoring:

```bash
python src/monitoring.py
```

Run final evaluation:

```bash
python src/final_evaluation.py
```

Generate reports:

```bash
python src/generate_reports.py
```

## 5. Validation Strategy

- Stratified training/validation/test split
- Five-fold cross-validation
- Randomized hyperparameter optimization
- Calibration using sigmoid and isotonic methods
- Threshold optimization on validation data
- Final test evaluation after model selection

The test set is not used for tuning or calibration.

## 6. Business Cost Assumptions

Fraud approved: LKR 5000

Manual review: LKR 100

Legitimate transaction blocked: LKR 500

Fraud blocked: LKR 20

Review detection rate: 80%

These are hypothetical assumptions.

## 7. Limitations

- Small dataset
- No exact transaction timestamps
- Synthetic distribution-shift experiments
- Hypothetical financial cost assumptions
- No real production fraud analyst feedback

## 8. Reproducibility

Random seed: 42

Data splits and trained artifacts are stored in outputs/models.

