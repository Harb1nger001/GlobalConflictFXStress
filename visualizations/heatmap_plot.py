import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_model_prediction_heatmap(pred_path: str, features_path: str, output_dir: str, model_name: str):
    """
    Plots a choropleth heatmap for predicted stress levels by country.

    Parameters:
    - pred_path (str): Path to CSV file containing model predictions (no country column)
    - features_path (str): Path to features CSV with 'country' column
    - output_dir (str): Folder to save the heatmap
    - model_name (str): Used for file naming and title
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load world map
    world = gpd.read_file(
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
    )
    world["ADMIN"] = world["ADMIN"].str.strip()

    # Load predictions and features
    preds = pd.read_csv(pred_path)
    features = pd.read_csv(features_path)

    # Add country names to predictions
    preds = preds.copy()
    preds["country"] = features["country"].str.strip()

    # Rename prediction column
    pred_col = preds.columns[0]
    preds.rename(columns={pred_col: "predicted_stress"}, inplace=True)

    # Aggregate predictions per country
    stress_map = preds.groupby("country")["predicted_stress"].mean().reset_index()

    # Merge with world geometry
    merged = world.merge(stress_map, how="left", left_on="ADMIN", right_on="country")

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 9))
    merged.plot(
        column="predicted_stress",
        ax=ax,
        cmap="coolwarm",
        legend=True,
        legend_kwds={"label": f"{model_name} Predicted Stress", "shrink": 0.6}
    )
    ax.set_title(f"🌍 Heatmap of Predicted Currency Stress — {model_name}", fontsize=16)
    ax.axis("off")

    # Save figure
    out_path = os.path.join(output_dir, f"{model_name.lower()}_stress_heatmap.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved {model_name} heatmap to: {out_path}")
