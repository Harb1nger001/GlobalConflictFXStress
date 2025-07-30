# fx_conflict_stress/features/macro_features.py

import pandas as pd
import os

def load_fx_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, skiprows=4)
    except Exception as e:
        raise RuntimeError(f"Failed to load FX rates data: {e}")

    if "Country Name" not in df.columns:
        raise ValueError("Expected 'Country Name' in FX rates data.")

    df = df.rename(columns={"Country Name": "country"})
    year_cols = [col for col in df.columns if col.strip().isdigit()]
    df = df.melt(id_vars="country", value_vars=year_cols, var_name="year", value_name="fx_rate")
    df.dropna(inplace=True)
    df["year"] = df["year"].astype(int)
    df["fx_rate"] = pd.to_numeric(df["fx_rate"], errors="coerce")
    df.dropna(subset=["fx_rate"], inplace=True)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    return df[["country", "date", "year", "fx_rate"]]

def load_reserves_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Country Name" not in df.columns:
        raise ValueError("Expected 'Country Name' in World Bank reserves data.")

    df = df.rename(columns={"Country Name": "country"})
    year_cols = [col for col in df.columns if col.strip().isdigit()]
    df = df.melt(id_vars="country", value_vars=year_cols, var_name="year", value_name="reserves")
    df.dropna(subset=["reserves"], inplace=True)
    df["year"] = df["year"].astype(int)
    df["reserves"] = pd.to_numeric(df["reserves"], errors="coerce")
    df.dropna(subset=["reserves"], inplace=True)
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    return df[["country", "date", "year", "reserves"]]

def load_fdi_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Country Name" not in df.columns:
        raise ValueError("FDI file must contain a 'Country Name' column.")

    df = df.rename(columns={"Country Name": "country"})
    year_cols = [col for col in df.columns if col.strip().isdigit()]
    df = df.melt(id_vars="country", value_vars=year_cols, var_name="year", value_name="fdi_pct_gdp")
    df.dropna(inplace=True)
    df["year"] = df["year"].astype(int)
    df["fdi_pct_gdp"] = pd.to_numeric(df["fdi_pct_gdp"], errors="coerce")
    df.dropna(subset=["fdi_pct_gdp"], inplace=True)
    return df[["country", "year", "fdi_pct_gdp"]]

def compute_macro_features(fx_df, reserves_df, fdi_df):
    fx_df = fx_df.sort_values(by=["country", "date"])
    reserves_df = reserves_df.sort_values(by=["country", "date"])

    fx_df["fx_return"] = fx_df.groupby("country")["fx_rate"].pct_change()
    fx_df["fx_drawdown"] = fx_df.groupby("country")["fx_rate"].transform(lambda x: (x / x.cummax()) - 1)

    reserves_df["reserves_return"] = reserves_df.groupby("country")["reserves"].pct_change()
    reserves_df["reserves_drawdown"] = reserves_df.groupby("country")["reserves"].transform(lambda x: (x / x.cummax()) - 1)

    print("✔ FX year range:", fx_df["year"].min(), "to", fx_df["year"].max())
    print("✔ Reserves year range:", reserves_df["year"].min(), "to", reserves_df["year"].max())
    print("✔ FDI year range:", fdi_df["year"].min(), "to", fdi_df["year"].max())
    print("✔ FX countries:", fx_df["country"].nunique())
    print("✔ Reserves countries:", reserves_df["country"].nunique())
    print("✔ FDI countries:", fdi_df["country"].nunique())

    merged = fx_df.merge(reserves_df, on=["country", "date", "year"], how="inner")
    print("🔍 Merged FX + Reserves rows:", len(merged))

    merged = merged.merge(fdi_df, on=["country", "year"], how="left")
    print("🔍 Final merged rows with FDI:", len(merged))

    return merged

def generate_macro_feature_panel(fx_path, reserves_path, fdi_path, output_path):
    fx_df = load_fx_data(fx_path)
    reserves_df = load_reserves_data(reserves_path)
    fdi_df = load_fdi_data(fdi_path)

    print("✔ Loaded FX data:")
    print(fx_df.head())
    print("✔ Loaded Reserves data:")
    print(reserves_df.head())
    print("✔ Loaded FDI data:")
    print(fdi_df.head())

    macro_df = compute_macro_features(fx_df, reserves_df, fdi_df)
    macro_df.to_csv(output_path, index=False)
    print(f"✅ Macro feature panel saved to {output_path}")

if __name__ == "__main__":
    base_dir = r"D:\\Global Conflict Currency Stress"
    fx_path = os.path.join(base_dir, "data", "raw", "API_PA.NUS.FCRF_DS2_en_csv_v2_22859.csv")
    reserves_path = os.path.join(base_dir, "data", "raw", "API_FI.RES.TOTL.CD_DS2_en_csv_v2_38110.csv")
    fdi_path = os.path.join(base_dir, "data", "raw", "API_BX.KLT.DINV.WD.GD.ZS_DS2_en_csv_v2_38342.csv")
    output_path = os.path.join(base_dir, "data", "processed", "macro_features.csv")

    generate_macro_feature_panel(fx_path, reserves_path, fdi_path, output_path)