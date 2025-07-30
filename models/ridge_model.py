import os
import joblib  # for saving sklearn models
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd


class RidgeStressModel:
    def __init__(self):
        self.model = RidgeClassifier()
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")

    def preprocess_features(self, X):
        if isinstance(X, pd.DataFrame):
            X_clean = X.select_dtypes(include=[np.number])
        elif isinstance(X, np.ndarray):
            X_clean = X
        else:
            raise ValueError("X must be either a pandas DataFrame or numpy ndarray.")

        if pd.isnull(X_clean).any().any():
            print("⚠️ Detected NaNs, applying imputation...")
        X_clean = self.imputer.fit_transform(X_clean)
        X_scaled = self.scaler.fit_transform(X_clean)
        return X_scaled

    def train(self, X_train, y_train):
        print(f"🔁 Training Ridge Regression on shape {X_train.shape}...")
        X_clean = self.preprocess_features(X_train)
        y_clean = y_train
        self.model.fit(X_clean, y_clean)

    def evaluate(self, X_test, y_test, results_dir="results", model_name="ridge"):
        X_clean = self.preprocess_features(X_test)
        y_pred = self.model.predict(X_clean)

        print("📊 Evaluation Results:\n")
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        df_report = pd.DataFrame(report).T
        print(df_report)

        # Ensure results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Save classification report
        report_path = os.path.join(results_dir, f"{model_name}_report.csv")
        df_report.to_csv(report_path)
        
        # Save predictions
        pred_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": y_pred
        })
        pred_path = os.path.join(results_dir, f"{model_name}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        # Save model
        model_path = os.path.join(results_dir, f"{model_name}_model.pkl")
        joblib.dump(self.model, model_path)

        print(f"✅ Ridge results saved to `{results_dir}/`")

        return report
