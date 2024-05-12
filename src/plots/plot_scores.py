import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


'''
This script produces the score plots for all 3 models and requires the score results csvs:
    output/lm_results.csv
    output/rf_results.csv
    output/nn_results.csv
The output plots will be produced at:
    output/lm_plot.png
    output/rf_plot.png
    output/nn_plot.png
'''

results_df = pd.read_csv('output/lm_results.csv')

# Create a figure and subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Plot contour plot for Ridge 
contour1 = axes[0].tricontourf(results_df[results_df['regularisation type']=='ridge']['shift'].values, 
                        np.log(results_df[results_df['regularisation type']=='ridge']['regularisation magnitude'].values),
                        results_df[results_df['regularisation type']=='ridge']['validation score'].values, cmap='viridis')
fig.colorbar(contour1, ax=axes[0], label='Mean score')
axes[0].set_xlabel('shift')
axes[0].set_ylabel('log alpha')
axes[0].set_title('Score vs shift and log alpha (Ridge)')

# Plot contour plot for Ridge 
contour2 = axes[1].tricontourf(results_df[results_df['regularisation type']=='lasso']['shift'].values, 
                        np.log(results_df[results_df['regularisation type']=='lasso']['regularisation magnitude'].values),
                        results_df[results_df['regularisation type']=='lasso']['validation score'].values, cmap='viridis')
fig.colorbar(contour2, ax=axes[1], label='Mean score')
axes[1].set_xlabel('shift')
axes[1].set_ylabel('log alpha')
axes[1].set_title('Score vs shift and log alpha (lasso)')

# Adjust layout
plt.tight_layout()

plt.savefig('output/lm_plot.png')
# Show the plot
plt.show()

'''
Producing plot for Random Forest
'''

results_df = pd.read_csv('output/rf_results.csv')

plt.figure(figsize=(12, 6))

true_df = results_df[results_df['bootstrap']==True].pivot(index='max depth', columns='shift', values='validation score')
plt.subplot(1, 2, 1)
sns.heatmap(true_df)
plt.title('Bootstrap')

false_df = results_df[results_df['bootstrap']==False].pivot(index='max depth', columns='shift', values='validation score')
plt.subplot(1, 2, 2)
sns.heatmap(false_df)
plt.title('Without Bootstrap')

plt.tight_layout()
plt.savefig('output/rf_plot.png')
plt.show()

'''
Producing plot for MLP/NN
'''

results_df = pd.read_csv('output/nn_results.csv')

plt.figure(figsize=(12, 6))

temp_df = results_df[results_df['number of hidden layers']==1].pivot(index='units per layer', columns='shift', values='validation score')
plt.subplot(1, 3, 1)
sns.heatmap(temp_df)
plt.title('1 Hidden Layer')

temp_df = results_df[results_df['number of hidden layers']==2].pivot(index='units per layer', columns='shift', values='validation score')
plt.subplot(1, 3, 2)
sns.heatmap(temp_df)
plt.title('2 Hidden Layer')

temp_df = results_df[results_df['number of hidden layers']==3].pivot(index='units per layer', columns='shift', values='validation score')
plt.subplot(1, 3, 3)
sns.heatmap(temp_df)
plt.title('3 Hidden Layer')

plt.tight_layout()
plt.savefig('output/nn_plot.png')
plt.show()