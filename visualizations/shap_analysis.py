import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np  # 👈 Needed for the RNG

def run_shap_analysis(model, X_train, feature_names, model_name="ridge", output_dir="shap_outputs"):
    """
    Perform SHAP analysis and save summary plots.

    Parameters:
        model: Trained model (must support shap)
        X_train: Feature DataFrame or ndarray
        feature_names: List of feature names (important for SHAP visualization)
        model_name: String identifier for model (used in filenames)
        output_dir: Directory where plots will be saved
    """

    os.makedirs(output_dir, exist_ok=True)

    # Convert to DataFrame if needed
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=feature_names)

    # Set up reproducible random number generator
    rng = np.random.default_rng(42)

    # Select SHAP explainer
    if model_name.lower() == "xgboost":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model.predict, X_train, seed=42)

    shap_values = explainer(X_train)

    # Save summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.title(f"{model_name.upper()} SHAP Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_shap_summary.png"))
    plt.close()

    print(f"✅ SHAP summary plot saved to: {os.path.join(output_dir, f'{model_name}_shap_summary.png')}")
