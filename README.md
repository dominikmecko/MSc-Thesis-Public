# MSc Thesis – Public Repository

This repository contains the code, data descriptions, and documentation for my Master of Science thesis, including analysis scripts, prediction models, evaluation diagnostics, and the final thesis document.

---

## 📄 Overview

**Title:** *[Insert your thesis title here]*  
**Author:** Dominik Mecko  
**Degree:** MSc  
**Institution:** [Your University]  
**Supervisor:** [Supervisor’s Name]  
**Year:** 2025 (or relevant)  

This repository supports the reproducible research associated with the thesis, including:
- Model training and evaluation code (Python / R)
- Data preprocessing pipelines
- Diagnostic plots and error analyses (e.g., calibration curves, error deciles)
- Final document in PDF

---

## 📁 Repository Structure

```text
MSc-Thesis-Public/
├── data/
│   ├── weather_processed.csv
│   ├── energy_eui.csv
│   └── README_DATA.md
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation_plots.ipynb
├── src/
│   ├── modeling.py
│   ├── metrics.py
│   ├── utils.py
│   └── conformal.py
├── results/
│   ├── calibration_plot.png
│   ├── error_by_decile.csv
│   └── prediction_intervals.csv
├── thesis/
│   ├── final_thesis.pdf
│   └── appendix.zip
├── environment.yml
├── requirements.txt
└── README.md
