from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

'''
This script produces the ACF and PACF plot of response variables for SGP and requires data/derived/cleaned_df.csv
It produces the plot at output/acf.png
'''

no_na_df = pd.read_csv('data/derived/cleaned_df.csv')

fig, ax = plt.subplots(nrows=6, ncols=1, figsize=(12, 8))
plot_acf(no_na_df.pivot(index='iso_code', columns='date', values='new_deaths_smoothed_per_million').loc['SGP'].values, 
         ax=ax[0], lags=15, title='Autocorrelation of new deaths smoothed per million')
plot_pacf(no_na_df.pivot(index='iso_code', columns='date', values='new_deaths_smoothed_per_million').loc['SGP'].values, 
          ax=ax[1], lags=15, title='Partial Autocorrelation of new deaths smoothed per million')

plot_acf(no_na_df.pivot(index='iso_code', columns='date', values='weekly_icu_admissions_per_million').loc['SGP'].values, 
         ax=ax[2], lags=15, title='Autocorrelation of ICU admissions per million')
plot_pacf(no_na_df.pivot(index='iso_code', columns='date', values='weekly_icu_admissions_per_million').loc['SGP'].values, 
         ax=ax[3], lags=15, title='Partial Autocorrelation of ICU admissions per million')

plot_acf(no_na_df.pivot(index='iso_code', columns='date', values='weekly_hosp_admissions_per_million').loc['SGP'].values, 
         ax=ax[4], lags=15, title='Autocorrelation of hospital admissions per million')
plot_pacf(no_na_df.pivot(index='iso_code', columns='date', values='weekly_hosp_admissions_per_million').loc['SGP'].values, 
          ax=ax[5], lags=15, title='Partial Autocorrelation of hospital admissions per million')

for i in range(6):
    ax[i].set_xlim([-0.5, 15.5])
    ax[i].set_ylim([-1.1, 1.1])
    
plt.tight_layout()
plt.savefig('output/acf.png')
plt.show()