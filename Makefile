SHELL := /bin/bash
default: article.pdf

data/derived/FracFocusRegistry_full.csv:
	# Collates all FracFocusRegitry_*.csv into 1
	python ./src/data_cleaning/collate_df.py

src/latex/images/disclosure_rules_cropped.jpg src/latex/images/fracked_states_cropped.jpg: data/derived/FracFocusRegistry_full.csv
	# Creates plots
	python ./buildplot.py

article.pdf: src/latex/images/disclosure_rules_cropped.jpg src/latex/images/fracked_states_cropped.jpg
	# Builds pdf and copies it onto directory level
	cd src/latex;\
		pdflatex article.tex;

	cp src/latex/article.pdf article.pdf

# Copy of an old makefile as an intial template
