# main.py

import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


from models.ridge_model import RidgeStressModel
from models.xgb_model import XGBoostStressModel
from models.rnn_model import RNNStressModel  # Now an RNN model, not GNN!
from visualizations.heatmap_plot import plot_model_prediction_heatmap
from visualizations.shap_analysis import run_shap_analysis


# === CONFIGURATION ===
DATA_DIR = "data/processed"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# === LOAD TABULAR DATA ===
print("📊 Loading tabular data...")
df = pd.read_csv(os.path.join(DATA_DIR, "features_and_labels.csv"))

# Drop rows with missing labels
df = df.dropna(subset=["stress_label"])

# Extract X and y
X = df.drop(columns=["stress_label"])
y = df["stress_label"]

# Ensure numeric features only
X = X.select_dtypes(include=["number"]).fillna(0)

# Label encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === MODEL 1: Ridge Regression ===
print("\n🔁 Training Ridge Regression...")
ridge_model = RidgeStressModel()
ridge_model.train(X, y_encoded)
ridge_model.evaluate(X, y_encoded)

# === MODEL 2: XGBoost ===
print("\n🌲 Training XGBoost...")
xgb_model = XGBoostStressModel()
xgb_model.train(X, y_encoded)
xgb_model.evaluate(X, y_encoded)

# === MODEL 3: RNN Model ===
print("\n🧠 Training RNN model...")
rnn_model = RNNStressModel(input_dim=X.shape[1], output_dim=len(le.classes_))
rnn_model.train(X, y_encoded)
rnn_model.evaluate(X, y_encoded)

# === PRINT FINAL EVALUATION SCORES ===
def print_results_table():
    print("\n📈 Final Evaluation Results:")
    for model_name in ["ridge", "xgboost", "rnn"]:
        path = os.path.join(RESULTS_DIR, f"{model_name}_report.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)  # 🛠️ Fixed!
            if "f1-score" in df.columns and "macro avg" in df.index:
                f1_macro = df.loc["macro avg", "f1-score"]
                print(f"✅ {model_name.upper():<8}: Macro F1 = {f1_macro:.4f}")
            else:
                print(f"⚠️ {model_name.upper()}: Missing 'macro avg' or 'f1-score'")
        else:
            print(f"❌ {model_name.upper()}: Report file not found")

print_results_table()

# === SHAP + HEATMAP ANALYSIS FOR INTERPRETABILITY ===
print("\n🔍 Running SHAP analysis for Ridge and XGBoost...")
run_shap_analysis(ridge_model.model, X, X.columns.tolist(), model_name="ridge")
run_shap_analysis(xgb_model.model, X, X.columns.tolist(), model_name="xgboost")

print("\n🔥 Plotting heatmaps of class distributions...")
plot_model_prediction_heatmap(
    pred_path=r"D:\Global Conflict Currency Stress\results\ridge_predictions.csv",
    features_path=r"D:\Global Conflict Currency Stress\data\processed\features_and_labels.csv",
    output_dir=r"D:\Global Conflict Currency Stress\plots",
    model_name="Ridge"
)

plot_model_prediction_heatmap(
    pred_path=r"D:\Global Conflict Currency Stress\results\xgboost_predictions.csv",
    features_path=r"D:\Global Conflict Currency Stress\data\processed\features_and_labels.csv",
    output_dir=r"D:\Global Conflict Currency Stress\plots",
    model_name="XGBoost"
)

print("\n✅ All done. Time for coffee ☕")
