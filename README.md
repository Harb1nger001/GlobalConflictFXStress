# 🌍 Global Conflict FX Stress Predictor

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-orange)

A machine learning pipeline that predicts currency stress events using macroeconomic indicators, geopolitical events, and military data. Designed for research on how global conflicts influence foreign exchange volatility.

---

## 🚀 Overview

This project integrates:
- 💹 Ridge Regression
- 🌲 XGBoost
- 🧠 RNN
- 🔍 SHAP Analysis
- 🗺️ Heatmap Visualizations

The pipeline is trained on a multiclass stress labeling system based on economic theory and empirical FX crisis indicators. It includes model evaluation, SHAP-based interpretability, and stress distribution mapping.

---

## 📁 Project Structure

```
Global Conflict Currency Stress/
│
├── data/                   # Raw and processed data (features, labels)
├── models/                 # Trained model weights
├── results/                # Model evaluation results
├── shap_outputs/           # SHAP summary plots
├── plots/                  # Heatmaps and visualizations
├── visualizations/         # Plotting and SHAP utilities
├── main.py                 # Master script
├── requirements.txt
└── README.md
```

---

## 🧪 Models Used

| Model      | Macro F1 Score |
|------------|----------------|
| Ridge      | 0.3197         |
| XGBoost    | 0.8322         |
| RNN        | 0.1079         |

SHAP plots show Ridge and XGBoost's feature importances. Heatmaps visualize stress severity by country and year.

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/global-conflict-fx-stress-predictor.git
cd global-conflict-fx-stress-predictor
pip install -r requirements.txt
python main.py
```

---

## 📊 Visual Output

- 🔥 `plots/` contains class-wise stress heatmaps
- 📈 `shap_outputs/` contains feature importance plots

---

## 🤝 Contributing

PRs are welcome! If you're into macroeconomic forecasting, political risk modeling, or quantum ML, this is your turf.

---

## 📜 License

This project is licensed under the MIT License.

---

## ☕ Acknowledgements

Shoutout to datasets from World Bank, ACLED, SIPRI, UCDP/PRIO, and others. Also, caffeine.
