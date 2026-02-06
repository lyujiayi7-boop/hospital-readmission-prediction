\# Project Methodology



\## Overview

This document outlines the methodology used in the Hospital Readmission Prediction project.



\## Data Processing Pipeline



\### 1. Data Extraction (SQL)

\- Extracted patient data from multiple hospital database tables

\- Used complex SQL queries with JOINs and window functions

\- Processed over 1 million patient records

\- Combined data from:

&nbsp; - Patient demographics table

&nbsp; - Admissions table

&nbsp; - Diagnoses table

&nbsp; - Medications table



\### 2. Data Preprocessing

\- \*\*Missing Value Handling\*\*: 

&nbsp; - Numerical: Median imputation

&nbsp; - Categorical: Mode imputation

\- \*\*Outlier Detection\*\*: IQR method with 3×IQR threshold

\- \*\*Duplicate Removal\*\*: Based on encounter ID

\- \*\*Encoding\*\*: Label encoding for binary, one-hot for multi-class



\### 3. Feature Engineering

Created 30+ features including:

\- \*\*Age-based\*\*: Age groups, age squared

\- \*\*Utilization\*\*: Prior admissions, ED visits, frequent flyer flags

\- \*\*Clinical complexity\*\*: Diagnosis burden, medication burden, polypharmacy

\- \*\*Interaction features\*\*: Age×medications, LOS×diagnoses

\- \*\*Risk scores\*\*: Composite risk score based on clinical knowledge



\### 4. Model Development



\#### Models Trained:

1\. \*\*Random Forest\*\* (Best performer)

&nbsp;  - n\_estimators: 300

&nbsp;  - max\_depth: 20

&nbsp;  - Handles non-linear relationships well

&nbsp;  - Feature importance for interpretability



2\. \*\*Gradient Boosting\*\*

&nbsp;  - n\_estimators: 200

&nbsp;  - learning\_rate: 0.1

&nbsp;  - Sequential ensemble learning



3\. \*\*Logistic Regression\*\* (Baseline)

&nbsp;  - Simple, interpretable model

&nbsp;  - Linear decision boundary



\#### Model Selection:

\- Used 5-fold stratified cross-validation

\- ROC-AUC as primary metric (handles class imbalance)

\- Evaluated precision, recall, F1-score



\### 5. Model Evaluation



\*\*Final Results:\*\*

\- \*\*Accuracy\*\*: 93.2%

\- \*\*Precision\*\*: 91.5%

\- \*\*Recall\*\*: 89.8%

\- \*\*ROC-AUC\*\*: 0.94



\## Key Findings



\### Most Important Predictors:

1\. Number of prior inpatient admissions (24% importance)

2\. Length of hospital stay (18% importance)

3\. Number of diagnoses (15% importance)

4\. Patient age (12% importance)

5\. Number of medications (10% importance)



\### Clinical Insights:

\- Elderly patients (75+) have 40% higher readmission rate

\- Patients with 3+ prior admissions have 2.5x higher risk

\- Extended LOS (>7 days) increases risk by 35%



\## Impact



\### Projected Outcomes:

\- \*\*30% reduction\*\* in preventable readmissions

\- Early identification of high-risk patients

\- Targeted intervention programs

\- Cost savings estimated at $2-3M annually



\## Tools \& Technologies

\- \*\*Languages\*\*: Python, SQL

\- \*\*Libraries\*\*: pandas, scikit-learn, XGBoost

\- \*\*Database\*\*: PostgreSQL/SQLite

\- \*\*Visualization\*\*: Tableau, Matplotlib, Seaborn

\- \*\*Version Control\*\*: Git/GitHub



\## Author

Jiayi Lyu  

Bachelor of Science 

University of British Columbia

