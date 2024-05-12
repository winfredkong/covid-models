import pandas as pd

'''
This script contains helpful functions/variables to prepare data for different choices
of hyperparameters later.
'''

time_dep_variables = ['people_fully_vaccinated_per_hundred', 'new_cases_smoothed_per_million', 'tests_per_case',
            'new_vaccinations_smoothed_per_million', 'stringency_index']
response_variables = ['new_deaths_smoothed_per_million', 'weekly_icu_admissions_per_million', 'weekly_hosp_admissions_per_million']
const_variables = ['population_density', 'median_age', 'aged_65_older',
                  'aged_70_older', 'gdp_per_capita', 'cardiovasc_death_rate', 'diabetes_prevalence',
                  'male_smokers', 'female_smokers', 'hospital_beds_per_thousand',
                  'life_expectancy', 'human_development_index']

# Define a function for shifting and splitting dataset
def add_shift_predictors(df, shift):
    '''
    Adds predictors based on shift and then drops all rows that are NA

    Inputs:
    df is the full dataframe with all the variables
    shift is an integer determining the number of days in the past to look at
    '''
    lag_df = df.copy()
    # Create lag features for predictors
    for feature in time_dep_variables+response_variables:
        for i in range(1, shift + 1):
            lag_df[f'{feature}_lag_{i}'] = df.groupby('iso_code')[feature].shift(i)
    lag_df = lag_df.dropna()

    return lag_df

# Define a function for splitting dataframe 
def split_xy(df):
    '''
    Splits dataframe into x and y

    Inputs:
    df is the full dataframe with all the variables with the shift predictors
    '''
    x = df.drop(columns=response_variables+['iso_code', 'date']+time_dep_variables)
    y = df[response_variables]
    return x, y