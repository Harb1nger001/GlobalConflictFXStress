# fx_conflict_stress/features/conflict_score.py

import pandas as pd
import os
from tqdm import tqdm

def load_ucdp_data(path: str) -> pd.DataFrame:
    """Load and preprocess UCDP conflict data."""
    df = pd.read_csv(path)
    df = df.rename(columns={'location': 'country'})
    return df[['year', 'country']]

def load_sipri_data(path: str) -> pd.DataFrame:
    """Load and preprocess SIPRI military expenditure data."""
    df = pd.read_csv(path, encoding='latin1')

    # Keep only country and year columns
    df = df.rename(columns={df.columns[0]: "country"})
    year_cols = [col for col in df.columns if str(col).isdigit()]
    df = df[["country"] + year_cols]

    # Reshape to long format
    df = df.melt(id_vars="country", var_name="year", value_name="mil_exp")
    df.dropna(inplace=True)
    df['year'] = df['year'].astype(int)
    df['mil_exp'] = pd.to_numeric(df['mil_exp'], errors='coerce')
    df.dropna(subset=['mil_exp'], inplace=True)
    return df

def compute_conflict_score(ucdp_df: pd.DataFrame, sipri_df: pd.DataFrame, country: str, year: int) -> float:
    """Compute a conflict score for a given country and year."""
    event_count = ucdp_df[(ucdp_df['country'] == country) & (ucdp_df['year'] == year)].shape[0]
    
    try:
        military_spending = sipri_df[(sipri_df['country'] == country) & (sipri_df['year'] == year)]['mil_exp'].values[0]
    except IndexError:
        military_spending = 0.0

    score = (0.7 * event_count) + (0.3 * military_spending / 1e9)
    return score

def generate_conflict_score_panel(ucdp_path: str, sipri_path: str, output_path: str, start_year: int = 2010, end_year: int = 2024):
    """Generate and save the conflict score panel."""
    ucdp_df = load_ucdp_data(ucdp_path)
    sipri_df = load_sipri_data(sipri_path)

    countries = sorted(set(ucdp_df['country']).intersection(set(sipri_df['country'])))
    years = range(start_year, end_year + 1)

    results = []
    for country in tqdm(countries, desc="Computing conflict scores"):
        for year in years:
            score = compute_conflict_score(ucdp_df, sipri_df, country, year)
            results.append({
                'country': country,
                'year': year,
                'conflict_score': score
            })

    score_df = pd.DataFrame(results)
    score_df.to_csv(output_path, index=False)
    print(f"✅ Conflict scores saved to {output_path}")

if __name__ == "__main__":
    base_dir = r"D:\\Global Conflict Currency Stress"
    ucdp_path = os.path.join(base_dir, "data", "raw", "UcdpPrioConflict_v25_1.csv")
    sipri_path = os.path.join(base_dir, "data", "raw", "SIPRI-Milex-data-1949-2024_2.csv")
    output_path = os.path.join(base_dir, "data", "processed", "conflict_score.csv")

    generate_conflict_score_panel(ucdp_path, sipri_path, output_path)
