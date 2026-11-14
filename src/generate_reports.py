
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs/reports"
TABLE_DIR = ROOT / "outputs/tables"
MODEL_DIR = ROOT / "outputs/models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

styles = getSampleStyleSheet()

title_style = styles["Title"]
heading_style = styles["Heading1"]
subheading_style = styles["Heading2"]
body_style = styles["BodyText"]

# -----------------------------------------
# LOAD MEASURED RESULTS
# -----------------------------------------

evaluation_path = TABLE_DIR / "final_test_evaluation.csv"

comparison_path = TABLE_DIR / "model_comparison.csv"

calibration_path = TABLE_DIR / "calibration_comparison.csv"

evaluation = pd.read_csv(evaluation_path).iloc[0]

comparison = pd.read_csv(comparison_path)

calibration = pd.read_csv(calibration_path)

selected_model = joblib.load(
    MODEL_DIR / "selected_model_name.joblib"
)

thresholds = joblib.load(
    MODEL_DIR / "selected_thresholds.joblib"
)


def add_table(data, widths=None):

    table = Table(
        data,
        colWidths=widths
    )

    table.setStyle(TableStyle([

        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E3A5F")),

        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),

        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),

        ("FONTSIZE", (0, 0), (-1, -1), 8),

        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

        ("PADDING", (0, 0), (-1, -1), 6)

    ]))

    return table


# =========================================
# MODEL CARD PDF
# =========================================

model_card_path = OUTPUT_DIR / "06_model_card.pdf"

doc = SimpleDocTemplate(
    str(model_card_path),
    pagesize=A4
)

story = []

story.append(Paragraph(
    "FRAUD RISK MODEL CARD",
    title_style
))

story.append(Spacer(1, 20))

story.append(Paragraph(
    "1. Model Overview",
    heading_style
))

story.append(Paragraph(
    "This model estimates the probability that a financial "
    "transaction is fraudulent. It supports operational "
    "decisions through Approve, Manual Review and Block actions.",
    body_style
))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "2. Intended Use",
    heading_style
))

story.append(Paragraph(
    "The intended use is fraud-risk prioritization for online "
    "and card transactions. The model is a decision-support "
    "system and should not be treated as an autonomous "
    "replacement for fraud investigation or governance.",
    body_style
))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "3. Training Dataset",
    heading_style
))

story.append(Paragraph(
    "Dataset: 3,000 transaction records. Target: fraud. "
    "Legitimate records: 2,636. Fraud records: 364. "
    "The dataset does not provide an exact transaction timestamp. "
    "Consequently, random stratified validation is used instead "
    "of genuine out-of-time validation.",
    body_style
))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "4. Selected Model",
    heading_style
))

story.append(Paragraph(
    f"Selected base model: {selected_model}. "
    "Selection is based on validation PR-AUC. "
    "Probability calibration is selected using validation "
    "Brier score.",
    body_style
))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "5. Evaluation Results",
    heading_style
))

metric_data = [

    ["Metric", "Final test result"],

    ["ROC-AUC", f"{evaluation['roc_auc']:.4f}"],

    ["PR-AUC", f"{evaluation['pr_auc']:.4f}"],

    ["Brier score", f"{evaluation['brier_score']:.4f}"],

    ["Fraud capture rate", f"{evaluation['fraud_capture_rate']:.4f}"],

    ["Review count", str(int(evaluation["review_count"]))],

    ["Block count", str(int(evaluation["block_count"]))]

]

story.append(add_table(metric_data))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "6. Operational Thresholds",
    heading_style
))

story.append(Paragraph(
    f"Review threshold: {thresholds['review_threshold']:.2f}. "
    f"Block threshold: {thresholds['block_threshold']:.2f}. "
    "These are validation-selected thresholds under the "
    "documented hypothetical cost assumptions.",
    body_style
))

story.append(Spacer(1, 12))

story.append(Paragraph(
    "7. Data Quality and Leakage Controls",
    heading_style
))

story.append(Paragraph(
    "Transaction identifiers and target labels are excluded "
    "from model features. Imputation, scaling, encoding and "
    "resampling are fitted inside the training pipeline. "
    "The absence of transaction timestamps limits temporal "
    "leakage analysis and time-based validation.",
    body_style
))

story.append(PageBreak())

story.append(Paragraph(
    "8. Limitations",
    heading_style
))

story.append(Paragraph(
    "The dataset contains only 3,000 records and may not "
    "represent real-world transaction distributions. Financial "
    "costs are hypothetical. Review detection effectiveness is "
    "assumed. Simulated distribution shifts do not establish "
    "future production performance. Geographic, device and "
    "customer-tier features may require fairness and compliance "
    "assessment before real deployment.",
    body_style
))

story.append(Spacer(1, 15))

story.append(Paragraph(
    "9. Monitoring Requirements",
    heading_style
))

story.append(Paragraph(
    "Monitor PR-AUC, precision, recall, Brier score, PSI, "
    "missing-value rates, review workload, block rate, "
    "prediction latency and expected business cost.",
    body_style
))

story.append(Spacer(1, 15))

story.append(Paragraph(
    "10. Retraining and Disable Policy",
    heading_style
))

story.append(Paragraph(
    "Review the model when labelled performance deteriorates, "
    "calibration becomes unreliable, important feature "
    "distributions shift, or operational workload exceeds "
    "capacity. Temporarily disable automated decisions if "
    "critical input fields are unavailable or predictions "
    "cannot be trusted.",
    body_style
))

story.append(Spacer(1, 15))

story.append(Paragraph(
    "11. Governance Status",
    heading_style
))

story.append(Paragraph(
    "This is an experimental internship project using a "
    "provided dataset and hypothetical business costs. "
    "It is not certified for real financial production deployment.",
    body_style
))

doc.build(story)

print("Model card created:", model_card_path)


# =========================================
# MANAGEMENT REPORT PDF
# =========================================

report_path = OUTPUT_DIR / "07_business_fraud_strategy.pdf"

doc = SimpleDocTemplate(
    str(report_path),
    pagesize=A4
)

story = []


def add_section(title, content):

    story.append(Paragraph(title, heading_style))

    story.append(Spacer(1, 8))

    story.append(Paragraph(content, body_style))

    story.append(Spacer(1, 15))


# PAGE 1
story.append(Paragraph(
    "BUSINESS FRAUD RISK STRATEGY",
    title_style
))

story.append(Spacer(1, 30))

add_section(
    "Executive Summary",
    "This report presents a fraud-risk intelligence pipeline "
    "designed to estimate transaction fraud probability, "
    "control false positives and support operational decisions."
)

story.append(PageBreak())

# PAGE 2
add_section(
    "1. Business Problem",
    "Fraud creates direct financial losses. Excessive false "
    "positives create legitimate customer friction. The proposed "
    "system balances both risks using calibrated probabilities "
    "and cost-sensitive decisions."
)

story.append(PageBreak())

# PAGE 3
add_section(
    "2. Dataset and Quality",
    "The dataset contains 3,000 transactions and 25 original "
    "columns. It contains 364 fraud transactions and 2,636 "
    "legitimate transactions. Missing values, extreme values, "
    "categorical cardinality and leakage risks are examined."
)

story.append(PageBreak())

# PAGE 4
add_section(
    "3. Exploratory Data Analysis",
    "EDA investigates fraud rates by channel, merchant category, "
    "authentication level, customer tier, device type and "
    "transaction time. Interaction analysis examines new-device "
    "distance and amount-merchant risk."
)

story.append(PageBreak())

# PAGE 5
add_section(
    "4. Feature Engineering",
    "Derived features include transaction velocity, amount "
    "normalization, combined risk scores, behavioural deviation, "
    "time indicators and interaction terms. Their availability "
    "at transaction decision time must be verified."
)

story.append(PageBreak())

# PAGE 6
story.append(Paragraph(
    "5. Model Performance",
    heading_style
))

comparison_data = [
    ["Model", "Validation PR-AUC"]
]

for _, row in comparison.iterrows():

    comparison_data.append([
        str(row["model"]),
        f"{row['validation_pr_auc']:.4f}"
    ])

story.append(add_table(comparison_data))

story.append(Spacer(1, 20))

add_section(
    "Selected Model",
    f"The selected model is {selected_model}, chosen using "
    "validation PR-AUC. Model selection also considers "
    "operational cost, calibration and complexity."
)

story.append(PageBreak())

# PAGE 7
story.append(Paragraph(
    "6. Calibration and Cost Optimization",
    heading_style
))

cal_data = [["Method", "Brier score"]]

for _, row in calibration.iterrows():

    cal_data.append([
        str(row["method"]),
        f"{row['brier_score']:.4f}"
    ])

story.append(add_table(cal_data))

story.append(Spacer(1, 20))

add_section(
    "Cost Assumptions",
    "The base scenario assumes LKR 5,000 fraud loss, LKR 100 "
    "manual review cost, LKR 500 false-block cost and LKR 20 "
    "fraud-block cost. Manual review is assumed to identify "
    "80% of fraud cases sent for review."
)

story.append(PageBreak())

# PAGE 8
story.append(Paragraph(
    "7. Operational Strategy",
    heading_style
))

story.append(Paragraph(
    f"Review threshold: {thresholds['review_threshold']:.2f}. "
    f"Block threshold: {thresholds['block_threshold']:.2f}.",
    body_style
))

story.append(Spacer(1, 20))

add_section(
    "Decision Framework",
    "Low-risk transactions are approved. Medium-risk "
    "transactions are sent to a human analyst. High-risk "
    "transactions are blocked under the proposed policy. "
    "Manual review capacity and customer appeal procedures "
    "must be established before production use."
)

story.append(PageBreak())

# PAGE 9
story.append(Paragraph(
    "8. Final Test Evidence",
    heading_style
))

test_data = [["Metric", "Value"]]

for metric in [
    "roc_auc",
    "pr_auc",
    "brier_score",
    "fraud_capture_rate",
    "default_policy_cost",
    "optimized_policy_cost"
]:

    test_data.append([
        metric,
        f"{evaluation[metric]:.4f}"
    ])

story.append(add_table(test_data))

story.append(Spacer(1, 20))

add_section(
    "9. Robustness",
    "Stress tests examine increased transaction amounts, "
    "higher velocity, missing-value changes, extreme values "
    "and channel distribution shifts. These are simulated "
    "scenarios and not substitutes for real future data."
)

story.append(PageBreak())

# PAGE 10
add_section(
    "10. Governance and Recommendations",
    "Monitor fraud detection performance, probability "
    "calibration, feature drift, score drift, missingness, "
    "review workload and business cost. Retrain only after "
    "reviewing new labelled evidence. Maintain a safe fallback "
    "procedure for system failures."
)

add_section(
    "Final Recommendation",
    "The proposed solution is a research and internship "
    "demonstration. Before real deployment, it requires "
    "larger representative datasets, chronological validation, "
    "real financial cost estimates, fairness assessment, "
    "security review, human oversight and operational testing."
)

doc.build(story)

print("Business report created:", report_path)
