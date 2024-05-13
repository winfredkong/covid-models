import pandas as pd
import numpy as np

'''
This script reads data/raw/owid-covid-data.csv and filters/cleans the data for further use
The resulting df is then stored in data/derived/cleaned_df.csv
'''

# Read csv
full_df = pd.read_csv('data/raw/owid-covid-data.csv')

# List all variables we will use for model
plot_cols = ['weekly_icu_admissions_per_million', 'new_deaths_smoothed_per_million', 'weekly_hosp_admissions_per_million', 
             'people_fully_vaccinated_per_hundred', 'new_cases_smoothed_per_million', 'tests_per_case',
             'new_vaccinations_smoothed_per_million', 'population_density', 'median_age', 'aged_65_older',
             'aged_70_older', 'gdp_per_capita', 'cardiovasc_death_rate', 'diabetes_prevalence', 
             'male_smokers', 'female_smokers', 'hospital_beds_per_thousand', 
             'life_expectancy', 'human_development_index', 'stringency_index']

# List of columns we will need from the dataframe
relevant_cols = ['iso_code', 'date']+plot_cols


'''
We note that there are some countries with NA in all values/dates of some columns. We may need to remove these countries from the analyses
for those variables since the numbers are not available (and not 0)

iso_all_na_df is a boolean df which contains True if index country has all na in a column value
'''

# Group by 'iso_code' and check if all values in specified columns are NA for each country
iso_all_na_df = full_df.groupby('iso_code')[full_df.columns].apply(lambda x: x.isna().all())


# Filter out the countries that record at least some information of all variables
countries_to_plot = iso_all_na_df.index[(iso_all_na_df[plot_cols]==False).all(axis=1)]
no_na_df = full_df[full_df['iso_code'].isin(countries_to_plot)][relevant_cols]


# List of constant predictor variables
constant_columns = ['population_density', 'median_age', 'aged_65_older',
                    'aged_70_older', 'gdp_per_capita', 'cardiovasc_death_rate', 'diabetes_prevalence',
                    'male_smokers', 'female_smokers', 'hospital_beds_per_thousand',
                    'life_expectancy', 'human_development_index']

'''
Some of the variables like life expectancy should be relatively constant and never 0. 
Thus we check that these entries are constant and if not we enforce them to be constant (if NA).
'''

# Iterate through each country and enforce constant values for constant variables
for iso_code, country_data in no_na_df.groupby('iso_code'):
    for column in constant_columns:
        mask = (country_data[column] == 0)
        if mask.all():  # If all values in the column are 0
            print(f'All data in {iso_code}, {column} is 0.')
            continue
        if mask.any():  # Check if any non-zero value exists
            constant_value = country_data[mask, column].iloc[0]  # Get the first non-zero value
            country_data.loc[~mask, column] = constant_value  # Fill 0 values with the constant value

        constant_value = country_data[column].iloc[0]  # Get the first value
        country_data.loc[:, column] = constant_value  # Set all values in the column to the constant value


'''
For cumulative time series variables it makes no sense to impute NA with 0. 
We interpolate all middle NA values and then forward fill the right hand side NAs,
then we fill left hand side NAs as 0.
'''

time_dependent_cols = ['people_fully_vaccinated_per_hundred',
                        'new_deaths_smoothed_per_million', 'weekly_icu_admissions_per_million', 'weekly_hosp_admissions_per_million',
                        'new_cases_smoothed_per_million', 'tests_per_case', 'new_vaccinations_smoothed_per_million', 'stringency_index']

# Function to interpolate missing values within each group
def interpolate_group(group):
    group['people_fully_vaccinated_per_hundred'] = group['people_fully_vaccinated_per_hundred'].interpolate().ffill().fillna(0)
    return group
    
# First note that cumulative (non-decreasing) data like fully_vaccinated cannot be filled with 0s in the middle and end
no_na_df = no_na_df.groupby('iso_code',group_keys=False).apply(interpolate_group)

# We don't want to fill a cumulative (non-decreasing) data like fully_vaccinated with 0s
no_na_df[time_dependent_cols[1:]] = no_na_df[time_dependent_cols[1:]].fillna(value=0)

no_na_df.to_csv('data/derived/cleaned_df.csv')
