# models/xgb_model.py

import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class XGBoostStressModel:
    def __init__(self, params=None):
        default_params = {
            "objective": "multi:softmax",
            "num_class": 3,         # Adjust based on your number of classes
            "eval_metric": "mlogloss",
            "use_label_encoder": False,
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42
        }
        self.params = params if params else default_params
        self.model = XGBClassifier(**self.params)
        self.encoder = None

    def preprocess_features(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X_clean = X.select_dtypes(include=[np.number])
        
        if X_clean.empty:
            raise ValueError("❌ No numeric features found in X.")

        X_clean = X_clean.fillna(X_clean.mean(numeric_only=True))
        
        return X_clean

    def train(self, X_train, y_train):
        X_clean = self.preprocess_features(X_train)
        self.model.fit(X_clean, y_train)

    def evaluate(self, X_test, y_test, results_dir="results", model_name="xgboost"):
        X_clean = self.preprocess_features(X_test)
        preds = self.model.predict(X_clean)

        report_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()

        os.makedirs(results_dir, exist_ok=True)
        report_path = os.path.join(results_dir, f"{model_name}_report.csv")
        report_df.to_csv(report_path)

        pred_df = pd.DataFrame({
            "y_true": y_test,
            "y_pred": preds
        })
        pred_path = os.path.join(results_dir, f"{model_name}_predictions.csv")
        pred_df.to_csv(pred_path, index=False)

        model_path = os.path.join(results_dir, f"{model_name}_model.pkl")
        joblib.dump(self.model, model_path)

        print(f"🌲 XGBoost results saved to: {results_dir}/")

    def predict(self, X):
        X_clean = self.preprocess_features(X)
        return self.model.predict(X_clean)
