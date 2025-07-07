import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === File Paths ===
botanical_path = r"C:\Users\HP\Desktop\Analysis 2\Botanical.xlsx"
inner_bark_path = r"C:\Users\HP\Desktop\Analysis 2\Data\inner_bark.xlsx"
forest_area_path = r"C:\Users\HP\Desktop\Analysis 2\Data\Forest_Area.xlsx"
ecoregion_path = r"C:\Users\HP\Desktop\Analysis 2\Data\Resolve_Ecoregions_-2451418960243700221.xlsx"
shapefile_path = r"C:\Users\HP\Documents\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp"
output_dir = r"C:\Users\HP\Desktop\Analysis 2\Figures"

os.makedirs(output_dir, exist_ok=True)

# === Utility Function ===
def save_plot(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    plt.close(fig)

# === Load Data ===
botanical_df = pd.read_excel(botanical_path)
inner_bark_df = pd.read_excel(inner_bark_path)
forest_df = pd.read_excel(forest_area_path)
ecoregion_df = pd.read_excel(ecoregion_path)
world = gpd.read_file(shapefile_path)

# === Data Cleaning ===
bad_values = ["...", "NA", "N/A", "-", "null", "Null", ""]

# Botanical cleaning
for col in ['oxidation rates/day', 'weight (g)', 'Incubation Days']:
    if col in botanical_df.columns:
        botanical_df[col] = pd.to_numeric(botanical_df[col].replace(bad_values, np.nan), errors='coerce')

# Inner bark cleaning and renaming
inner_bark_df.rename(columns={'Sample': 'Tree Type', 'Oxidation Rates /day': 'oxidation rates/day'}, inplace=True)
for col in ['oxidation rates/day', 'Incubation Days']:
    if col in inner_bark_df.columns:
        inner_bark_df[col] = pd.to_numeric(inner_bark_df[col].replace(bad_values, np.nan), errors='coerce')

botanical_df['Tissue'] = 'Outer Bark'
inner_bark_df['Tissue'] = 'Inner Bark'

# Combine datasets with consistent columns
combined_df = pd.concat([
    botanical_df[['Tree Type', 'oxidation rates/day', 'weight (g)', 'Incubation Days', 'Tissue']],
    inner_bark_df[['Tree Type', 'oxidation rates/day', 'Incubation Days', 'Tissue']]
], ignore_index=True)

combined_df['oxidation rates/day'] = pd.to_numeric(combined_df['oxidation rates/day'], errors='coerce').clip(lower=0)
combined_df.dropna(subset=['oxidation rates/day'], inplace=True)

# Assign random biome labels for demo (replace with real biome data if available)
combined_df['Biome'] = np.random.choice(['Tropical', 'Temperate', 'Boreal', 'Montane'], size=len(combined_df))

# Clean forest area data
forest_df['Country and Area'] = forest_df['Country and Area'].str.title().str.strip()
forest_melt = forest_df.melt(id_vars='Country and Area', value_vars=[
    'Forest Area, 1990', 'Forest Area, 2000', 'Forest Area, 2010', 'Forest Area, 2015', 'Forest Area, 2020'
], var_name='Year', value_name='Forest Area')
forest_melt['Year'] = forest_melt['Year'].str.extract(r'(\d{4})').astype(int)
forest_melt['Forest Area'] = pd.to_numeric(forest_melt['Forest Area'].replace(bad_values, np.nan), errors='coerce')

# === PLOTS ===
# 1. Boxplot by Tissue
fig = plt.figure(figsize=(8, 6))
sns.boxplot(data=combined_df, x='Tissue', y='oxidation rates/day')
plt.title("Oxidation Rates by Tissue Type")
save_plot(fig, "fig01_boxplot_tissue")

# 2. KDE
fig = plt.figure(figsize=(8, 6))
sns.kdeplot(data=combined_df, x='oxidation rates/day', hue='Tissue', fill=True)
plt.title("KDE: Oxidation Rates by Tissue")
save_plot(fig, "fig02_kde_tissue")

# 3. Violin Plot
fig = plt.figure(figsize=(8, 6))
sns.violinplot(data=combined_df, x='Tissue', y='oxidation rates/day')
plt.title("Violin Plot: Oxidation Rates")
save_plot(fig, "fig03_violin")

# 4. Biome Boxplot
fig = plt.figure(figsize=(10, 6))
sns.boxplot(data=combined_df, x='Biome', y='oxidation rates/day', hue='Tissue')
plt.title("Oxidation Rates by Biome and Tissue")
save_plot(fig, "fig04_biome_box")

# 5. Stripplot by Tree
fig = plt.figure(figsize=(14, 6))
sns.stripplot(data=combined_df, x='Tree Type', y='oxidation rates/day', hue='Tissue', jitter=True)
plt.xticks(rotation=90)
plt.title("Oxidation by Tree Type")
save_plot(fig, "fig05_tree_strip")

# 6. Biome Mean Bar Plot
biome_avg = combined_df.groupby(['Biome', 'Tissue'])['oxidation rates/day'].mean().unstack()
fig = biome_avg.plot(kind='bar', figsize=(10, 6), title='Mean Oxidation by Biome and Tissue').get_figure()
save_plot(fig, "fig06_biome_avg")

# 7. Forest Trends (Top 5 Countries by Mean Forest Area)
top5_countries = forest_melt.groupby('Country and Area')['Forest Area'].mean().nlargest(5).index
fig = plt.figure(figsize=(10, 6))
sns.lineplot(data=forest_melt[forest_melt['Country and Area'].isin(top5_countries)], x='Year', y='Forest Area', hue='Country and Area')
plt.title("Forest Area Trends (Top 5 Countries)")
save_plot(fig, "fig07_forest_trends")

# 8. Deforestation Histogram (2015-2020)
deforestation_data = pd.to_numeric(forest_df['Deforestation, 2015-2020'], errors='coerce').dropna()
fig = plt.figure(figsize=(8, 6))
sns.histplot(deforestation_data, bins=25, kde=True, color='tomato')
plt.title("Deforestation (2015–2020)")
save_plot(fig, "fig08_deforestation")

# 9. Pairplot of Botanical Variables
pair_data = botanical_df[['oxidation rates/day', 'weight (g)', 'Incubation Days']].dropna()
sns.pairplot(pair_data).fig.suptitle("Pairwise Relationships", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "fig09_pairplot.png"), dpi=300)
plt.close()

# 10. Correlation Heatmap
fig = plt.figure(figsize=(6, 5))
corr = pair_data.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
save_plot(fig, "fig10_corr")

# 11. Biome Box + Strip Plot
fig = plt.figure(figsize=(12, 6))
sns.boxplot(data=combined_df, x='Biome', y='oxidation rates/day', hue='Tissue', fliersize=0)
sns.stripplot(data=combined_df, x='Biome', y='oxidation rates/day', color='black', alpha=0.3, jitter=0.2)
plt.title("Box + Strip Plot")
save_plot(fig, "fig11_boxstrip")

# 12. Tree Type Bar Plot (mean oxidation)
tree_avg = combined_df.groupby('Tree Type')['oxidation rates/day'].mean().sort_values(ascending=False)
fig = plt.figure(figsize=(12, 6))
tree_avg.plot(kind='bar', color='slateblue')
plt.title("Oxidation Rate by Tree")
plt.ylabel("Rate")
plt.xticks(rotation=90)
save_plot(fig, "fig12_tree_bar")

# 13. Oxidation vs Incubation Scatter Plot
fig = plt.figure(figsize=(8, 6))
sns.scatterplot(data=combined_df, x='Incubation Days', y='oxidation rates/day', hue='Tissue')
plt.title("Oxidation vs Incubation")
save_plot(fig, "fig13_scatter")

# 14. Log Transformed Histogram
fig = plt.figure(figsize=(8, 6))
sns.histplot(np.log1p(combined_df['oxidation rates/day']), bins=30, color='seagreen')
plt.title("Log Oxidation Rate Distribution")
save_plot(fig, "fig14_log_hist")

print("✅ All plots saved.")

# === GLOBAL UPSCALING ===
oxidation_mean = combined_df['oxidation rates/day'].mean()
oxidation_std = combined_df['oxidation rates/day'].std()

# Convert oxidation rate per g/day to per m²/year (assuming 1g bark/m²)
mean_ug_m2_yr = oxidation_mean * 365
std_ug_m2_yr = oxidation_std * 365

# Global forest area in m² (2020)
global_forest_km2 = forest_melt[forest_melt['Year'] == 2020]['Forest Area'].sum()
global_forest_m2 = global_forest_km2 * 1e6  # km² to m²

# Total global oxidation Tg/yr
global_total_Tg = mean_ug_m2_yr * global_forest_m2 * 1e-12

print(f"🌍 Estimated Global CH₄ Oxidation (Tg/yr): {global_total_Tg:.3f}")

# === MONTE CARLO SIMULATION FOR GLOBAL UNCERTAINTY ===
n_sim = 10000
samples = np.random.normal(loc=oxidation_mean, scale=oxidation_std, size=n_sim)
samples = np.clip(samples, 0, None)  # No negative oxidation rates
total_oxidation_Tg_sim = samples * 365 * global_forest_m2 * 1e-12

ci_lower = np.percentile(total_oxidation_Tg_sim, 2.5)
ci_upper = np.percentile(total_oxidation_Tg_sim, 97.5)

print(f"95% Confidence Interval for Global Oxidation: {ci_lower:.3f} – {ci_upper:.3f} Tg/yr")

# Histogram of MC simulation
fig = plt.figure(figsize=(8,6))
sns.histplot(total_oxidation_Tg_sim, bins=50, kde=True, color='steelblue')
plt.axvline(global_total_Tg, color='red', linestyle='--', label='Mean Estimate')
plt.axvline(ci_lower, color='black', linestyle=':', label='95% CI Lower')
plt.axvline(ci_upper, color='black', linestyle=':', label='95% CI Upper')
plt.legend()
plt.title("Monte Carlo Simulation of Global Methane Oxidation (Tg/yr)")
save_plot(fig, "fig15_mc_simulation")

# === SAVE RESULTS TO EXCEL ===
results_df = pd.DataFrame({
    'Estimate_Tg_per_year': [global_total_Tg],
    'CI_lower_95%': [ci_lower],
    'CI_upper_95%': [ci_upper]
})

excel_outpath = r"C:\Users\HP\Desktop\Analysis 2\Global_Methane_Oxidation_Results.xlsx"
with pd.ExcelWriter(excel_outpath) as writer:
    combined_df.to_excel(writer, sheet_name='Combined Data', index=False)
    results_df.to_excel(writer, sheet_name='Global Oxidation Summary', index=False)

print(f"✅ Results and combined data saved to {excel_outpath}")
