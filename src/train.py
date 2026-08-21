import argparse
import json
from pathlib import Path
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.data import load_training_data
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

def build_pipeline() -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("preprocessor", preprocessor), ("model", LogisticRegression(max_iter=1_000, class_weight="balanced", n_jobs=-1))])

def train(data_path: str, output_dir: str) -> None:
    X, y = load_training_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    probability = pipeline.predict_proba(X_test)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    metrics = {"rows": len(X), "default_rate": float(y.mean()), "roc_auc": float(roc_auc_score(y_test, probability)), "average_precision": float(average_precision_score(y_test, probability)), "classification_report": classification_report(y_test, prediction, output_dict=True)}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "credit_risk_pipeline.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Kaggle loan.csv")
    parser.add_argument("--output", default="artifacts")
    args = parser.parse_args()
    train(args.data, args.output)
