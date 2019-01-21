SHELL=/bin/bash

all: create actograms episodes summary waveform 

create:
	@echo "activating env"
	@echo "creating files"
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/01_createfiles; \
	ls *.py; \
	python 03_clean.py; \
	python 04_create_sleep.py; \
	python 05_cleansleep.py

actograms:
	@echo "activating env"
	@echo "running actograms"; \
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/02_activity/01_actograms; \
	ls *.py; \
	python 01_longactogram.py; \
	python 02_shortactogram.py; \


episodes:
	@echo "activating env"
	@echo "running episodes"; \
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/02_activity/02_episodes; \
	ls *.py; \
	python 01_create_files.py; \
	python 02_histograms.py; \
	python 04_histogram_sum.py


summary:
	@echo "activating env"
	@echo "running summaries"; \
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/02_activity/03_summary_stats; \
	python *.py

waveform:
	@echo "activating env"
	@echo "running waveform"; \
	source activate actiPy_environment; \
	cd /Users/angusfisk/Documents/01_PhD_files/01_projects/01_thesisdata/02_circdis/02_analysis_files/02_activity/04_mean_waveform; \
	python *.py




