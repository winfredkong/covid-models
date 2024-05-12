import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
This script produces the heatmap plot of all relevant variables and requires data/derived/cleaned_df.csv
It produces 2 heatmap plots at output/heatmap_plot1.png and output/heatmap_plot2.png
'''

no_na_df = pd.read_csv('data/derived/cleaned_df.csv')

# Variables to plot for heatmap
plot_cols = ['weekly_icu_admissions_per_million', 'new_deaths_smoothed_per_million', 'weekly_hosp_admissions_per_million', 
             'people_fully_vaccinated_per_hundred', 'new_cases_smoothed_per_million', 'tests_per_case',
             'new_vaccinations_smoothed_per_million', 'population_density', 'median_age', 'aged_65_older',
             'aged_70_older', 'gdp_per_capita', 'cardiovasc_death_rate', 'diabetes_prevalence', 
             'male_smokers', 'female_smokers', 'hospital_beds_per_thousand', 
             'life_expectancy', 'human_development_index', 'stringency_index']

# Plot first heatmap
img_path = 'output/heatmap_plot1.png'

# Heat Plotting all columns
plt.figure(figsize=(9, 12))
for i, col in enumerate(plot_cols[:12]):
    temp_df = no_na_df.pivot(index='iso_code', columns='date', values=col).fillna(0)
    ax = plt.subplot(3, 4, i+1)
    sns.heatmap(temp_df, ax=ax)
    ax.set_title(col, fontsize=6)  # Increase title font size
    plt.xticks(size=6)  # Rotate x-axis labels
plt.tight_layout(pad=2.0)  # Increase spacing between subplots
plt.savefig(img_path)
plt.show()

# Plot second heatmap
img_path = 'output/heatmap_plot2.png'

# Heat Plotting all columns
plt.figure(figsize=(9, 12))
for i, col in enumerate(plot_cols[12:]):
    temp_df = no_na_df.pivot(index='iso_code', columns='date', values=col).fillna(0)
    ax = plt.subplot(2, 4, i+1)
    sns.heatmap(temp_df, ax=ax)
    ax.set_title(col, fontsize=6)  # Increase title font size
    plt.xticks(size=6)  # Rotate x-axis labels
plt.tight_layout(pad=2.0)  # Increase spacing between subplots
plt.savefig(img_path)
plt.show()
    