import os
import torch
import pandas as pd
import joblib
import numpy as np

from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

class RNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.2):
        super(RNNModel, self).__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)  # out: [batch_size, seq_len, hidden_dim]
        out = out[:, -1, :]    # grab last time step
        return self.fc(out)

class RNNStressModel:
    def __init__(self, input_dim, hidden_dim=64, output_dim=3, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RNNModel(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = CrossEntropyLoss()
        self.feature_encoders = {}

    def encode_features(self, X):
        X_encoded = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                self.feature_encoders[col] = le
        return X_encoded

    def encode_labels(self, y):
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            return torch.tensor(y_encoded, dtype=torch.long)
        return torch.tensor(y.values if isinstance(y, pd.Series) else y, dtype=torch.long)

    def train(self, X, y, epochs=100, batch_size=32):
        self.model.train()
        X = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                print(f"🧠 Epoch {epoch} | Loss: {loss.item():.4f}")

    def evaluate(self, X, y, results_dir="results", model_name="rnn"):
        self.model.eval()
        X = torch.tensor(X.values, dtype=torch.float32).unsqueeze(1).to(self.device)
        y = torch.tensor(y, dtype=torch.long).to(self.device)

        with torch.no_grad():
            output = self.model(X)
            preds = output.argmax(dim=1).cpu()
            y_true = y.cpu()

            report = classification_report(y_true, preds, output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()

            os.makedirs(results_dir, exist_ok=True)
            report_df.to_csv(os.path.join(results_dir, f"{model_name}_report.csv"), index=True)

            pred_df = pd.DataFrame({
                "y_true": y_true.numpy(),
                "y_pred": preds.numpy()
            })
            pred_df.to_csv(os.path.join(results_dir, f"{model_name}_predictions.csv"), index=False)

            torch.save(self.model.state_dict(), os.path.join(results_dir, f"{model_name}_model.pt"))
            print(f"✅ RNN results saved to `{results_dir}/`")

    def predict(self, X):
        self.model.eval()
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        X = X.unsqueeze(1).to(self.device)  # Add seq_len=1
        with torch.no_grad():
            output = self.model(X)
        return output.argmax(dim=1).cpu()
