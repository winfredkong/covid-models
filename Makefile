SHELL := /bin/bash
default: document.pdf

data/derived/cleaned_df.csv: 
# Cleans Data and filters it for use
	python src/data_cleaning/cleaning.py

src/latex/tables/adfstat.csv: data/derived/cleaned_df.csv
# Computes ADF stat and p-values during data exploration
	python src/main/compute_adf.py
	cp output/adfstat.csv src/latex/tables/adfstat.csv

output/lm_results.csv output/rf_results.csv output/nn_results.csv: data/derived/cleaned_df.csv
# Trains and collates validation scores of all 3 models
	python src/main/train.py

src/latex/images/acf.png: data/derived/cleaned_df.csv
# Produces acf and pacf plots during data exploration
	python src/plots/acf.py
	cp output/acf.png src/latex/images/acf.png

src/latex/images/heatmap_plot1.png src/latex/images/heatmap_plot2.png: data/derived/cleaned_df.csv
# Produces plots of heatmaps of variables during data exploration
	python src/plots/heatmap.py
	cp output/heatmap_plot1.png src/latex/images/heatmap_plot1.png
	cp output/heatmap_plot2.png src/latex/images/heatmap_plot2.png

src/latex/images/lm_plot.png src/latex/images/rf_plot.png  src/latex/images/nn_plot.png src/latex/tables/test_scores.csv: output/lm_results.csv\
output/rf_results.csv output/nn_results.csv
# Produces plots of scores of different model during validation/train phase
	python src/plots/plot_scores.py
	cp output/lm_plot.png src/latex/images/lm_plot.png
	cp output/rf_plot.png src/latex/images/rf_plot.png
	cp output/nn_plot.png src/latex/images/nn_plot.png
	cp output/tet_scores.csv src/latex/tables/test_scores.csv

document.pdf: src/latex/images/acf.png src/latex/images/heatmap_plot1.png src/latex/images/heatmap_plot2.png \
src/latex/images/lm_plot.png src/latex/images/rf_plot.png  src/latex/images/nn_plot.png\
src/latex/tables/adfstat.csv src/latex/tables/test_scores.csv
# Builds final pdf and makes a copy at project level
	cd src/latex;\
		pdflatex document.tex;
	
	cp src/latex/document.pdf document.pdf

