# fx_conflict_stress/features/generate_target_labels.py

import pandas as pd
import numpy as np
import os

def load_macro_features(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df["year"] = df["year"].astype(int)
    return df

def label_fx_stress(df, threshold=0.20):
    df = df.sort_values(by=["country", "year"])
    df["fx_change"] = df.groupby("country")["fx_rate"].pct_change()
    df["fx_stress"] = (df["fx_change"] > threshold).astype(int)
    return df

def label_reserves_stress(df, threshold=0.15):
    df = df.sort_values(by=["country", "year"])
    df["reserves_change"] = df.groupby("country")["reserves"].pct_change()
    df["reserves_stress"] = (df["reserves_change"] < -threshold).astype(int)
    return df

def label_fdi_stress(df, threshold=0.20):
    df = df.sort_values(by=["country", "year"])
    df["fdi_change"] = df.groupby("country")["fdi_pct_gdp"].pct_change(fill_method=None)
    df["fdi_stress"] = (df["fdi_change"] < -threshold).astype(int)
    return df

def load_conflict_data(path):
    df = pd.read_excel(path)
    df = df.rename(columns={"Country": "country", "Year": "year", "Events": "conflict_events"})
    df = df[df["year"].notna()]
    df["year"] = df["year"].astype(int)
    df["conflict"] = (df["conflict_events"] > 50).astype(int)
    return df[["country", "year", "conflict"]]

def label_alliance_conflict(alliance_path):
    df = pd.read_csv(alliance_path)
    df = df.rename(columns={"state_name1": "country1", "state_name2": "country2"})
    df = df[(df["defense"] == 1) | (df["entente"] == 1) | (df["nonaggression"] == 1)]

    df["start_year"] = df["dyad_st_year"]
    df["end_year"] = df["dyad_end_year"].fillna(2100).astype(int)

    records = []
    for _, row in df.iterrows():
        for year in range(row["start_year"], row["end_year"]):
            records.append({"country": row["country1"], "year": year})
            records.append({"country": row["country2"], "year": year})

    full_df = pd.DataFrame(records).drop_duplicates()
    full_df["alliance"] = 1
    full_df["key"] = full_df["country"] + "_" + full_df["year"].astype(str)

    end_df = df[df["dyad_end_year"].notna()]
    records_end = []
    for _, row in end_df.iterrows():
        records_end.append({"country": row["country1"], "year": int(row["dyad_end_year"])} )
        records_end.append({"country": row["country2"], "year": int(row["dyad_end_year"])} )

    lost_df = pd.DataFrame(records_end).drop_duplicates()
    lost_df["lost_alliance"] = 1
    return lost_df

def merge_all_labels(macro_df, conflict_df, alliance_lost_df):
    df = macro_df.merge(conflict_df, on=["country", "year"], how="left")
    df = df.merge(alliance_lost_df, on=["country", "year"], how="left")
    df["conflict"] = df["conflict"].fillna(0)
    df["lost_alliance"] = df["lost_alliance"].fillna(0)

    # Optional: define a combined multiclass label (e.g., 0–5 based on how many stressors exist)
    df["stress_score"] = df[["fx_stress", "reserves_stress", "fdi_stress", "conflict", "lost_alliance"]].sum(axis=1)
    return df

def save_labels(df, output_path):
    df_out = df[["country", "year", "fx_stress", "reserves_stress", "fdi_stress", "conflict", "lost_alliance", "stress_score"]]
    df_out.to_csv(output_path, index=False)


def main():
    base_dir = r"D:\\Global Conflict Currency Stress"
    macro_path = os.path.join(base_dir, "data", "processed", "macro_features.csv")
    conflict_path = os.path.join(base_dir, "data", "raw", "number_of_political_violence_events_by_country-month-year_as-of-04Jul2025.xlsx")
    alliance_path = os.path.join(base_dir, "data", "raw", "alliance_v4.1_by_dyad_yearly.csv")
    output_path = os.path.join(base_dir, "data", "processed", "target_labels.csv")

    macro_df = load_macro_features(macro_path)
    macro_df = label_fx_stress(macro_df)
    macro_df = label_reserves_stress(macro_df)
    macro_df = label_fdi_stress(macro_df)

    conflict_df = load_conflict_data(conflict_path)
    alliance_lost_df = label_alliance_conflict(alliance_path)

    full_df = merge_all_labels(macro_df, conflict_df, alliance_lost_df)
    save_labels(full_df, output_path)

    print(f"✅ Labels saved to {output_path}")

if __name__ == "__main__":
    main()
