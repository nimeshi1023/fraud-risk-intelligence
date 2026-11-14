
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

files_to_copy = {

    ROOT / "data/processed/02_processed_dataset.csv":
        ROOT / "02_processed_dataset.csv",

    ROOT / "outputs/models/03_fraud_model_pipeline.joblib":
        ROOT / "03_fraud_model_pipeline.joblib",

    ROOT / "outputs/models/04_calibrated_model.joblib":
        ROOT / "04_calibrated_model.joblib",

    ROOT / "outputs/tables/05_threshold_cost_analysis.csv":
        ROOT / "05_threshold_cost_analysis.csv",

    ROOT / "outputs/reports/06_model_card.pdf":
        ROOT / "06_model_card.pdf",

    ROOT / "outputs/reports/07_business_fraud_strategy.pdf":
        ROOT / "07_business_fraud_strategy.pdf",

    ROOT / "README.md":
        ROOT / "08_README.md"

}

for source, destination in files_to_copy.items():

    if source.exists():

        shutil.copy2(
            source,
            destination
        )

        print("Copied:", destination.name)

    else:

        print("MISSING:", source)


print("\nPackaging completed!")
