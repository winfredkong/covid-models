from statsmodels.tsa.stattools import adfuller
import pandas as pd

'''
This script computes the adf statistic and p-values for all responses (for all iso codes) and requires data/derived/cleaned_df.csv
The csv is produces at output/adfstat.csv
'''

no_na_df = pd.read_csv('data/derived/cleaned_df.csv')

def get_stationary(series):
    '''
    Returns the p-value of ADF statistic for an input series
    '''
    result = adfuller(series.values)

    return result[1]

#'new_deaths_smoothed_per_million', 'weekly_icu_admissions_per_million', 'weekly_hosp_admissions_per_million'
stationary_death = no_na_df.groupby('iso_code').apply(lambda group: get_stationary(group['new_deaths_smoothed_per_million']))
stationary_icu = no_na_df.groupby('iso_code').apply(lambda group: get_stationary(group['weekly_icu_admissions_per_million']))
stationary_hosp = no_na_df.groupby('iso_code').apply(lambda group: get_stationary(group['weekly_hosp_admissions_per_million']))

adf_df = pd.concat([stationary_death, stationary_icu, stationary_hosp], axis=1).rename(columns={0: 'p-value of death', 1: 'p-values of ICU admissions', 2: 'p-values of hospital admissions'})
adf_df.index.name='ISO code'
adf_df.to_csv('output/adfstat.csv')