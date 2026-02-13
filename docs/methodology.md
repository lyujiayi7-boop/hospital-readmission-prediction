# Project Methodology

## Background

This project started as an exploration into how statistical learning could be applied to healthcare data. The goal was to build a system that could identify patients at high risk of hospital readmission within 30 days of discharge.

---

## 1. Data Collection & Extraction

### Source Data
I worked with a synthetic dataset simulating real hospital records (1.1M+ patient admissions). In a real-world scenario, this would come from Electronic Health Records (EHR) systems.

### SQL Extraction Pipeline
Built SQL queries to extract and join data from multiple tables:
- Patient demographics (age, gender, race)
- Admission records (dates, admission type, length of stay)
- Diagnoses (ICD codes, number of diagnoses)
- Medications (drug counts, changes)

**Challenge**: The data was messy - lots of missing values, inconsistent formats, and duplicates. Had to build validation checks to ensure data quality.

---

## 2. Data Preprocessing

### Handling Missing Data
Tried different strategies:
- For numerical features (age, LOS): Used median imputation
- For categorical features: Mode imputation
- Dropped records with >30% missing values

### Outlier Detection
Used IQR (Interquartile Range) method:
- Anything beyond Q1 - 3×IQR or Q3 + 3×IQR flagged as outlier
- Investigated outliers rather than blindly removing (some were legitimate edge cases)

### Data Validation
- Checked for logical inconsistencies (e.g., discharge before admission)
- Verified distributions matched expected patterns
- Removed ~2% of records due to quality issues

---

## 3. Feature Engineering

This was the most interesting part. Created 50+ features based on:

### Statistical Features
- **Polynomial terms**: Age², LOS² (to capture non-linear relationships)
- **Interaction terms**: Age×Medications, LOS×Diagnoses
- **Ratios**: Emergency visit ratio, readmission history ratio

### Clinical Features
- Prior admission count (inpatient, outpatient, emergency)
- Medication burden (number of medications)
- Diagnosis complexity (number of diagnoses)
- "Frequent flyer" flags (3+ admissions in past year)

### Composite Scores
Built a risk score combining multiple factors:
```python
risk_score = 0.3 * prior_admissions + 0.2 * age_normalized + 
             0.2 * los_normalized + 0.15 * num_diagnoses + 
             0.15 * num_medications
```

**Validation**: Used chi-square tests and t-tests to verify that features were actually associated with readmission outcomes.

---

## 4. Model Development

### Why Ensemble Methods?
Started with Logistic Regression as baseline, but healthcare data has complex non-linear patterns. Ensemble methods (Random Forest, Gradient Boosting) handle this better.

### Models Trained

**Random Forest** (Primary Model)
- 300 trees, max depth 20
- Class-weighted to handle imbalance (70% no readmission, 30% readmission)
- Provides feature importance rankings

**Gradient Boosting**
- 200 estimators, learning rate 0.1
- Sequential learning - each tree corrects previous errors
- Slightly better precision but lower recall

**Logistic Regression** (Baseline)
- Simple linear model for comparison
- Helps understand if complexity is needed

### Hyperparameter Tuning
Used 5-fold cross-validation to avoid overfitting. Tried different combinations but avoided exhaustive grid search (computationally expensive for 1M+ records).

---

## 5. Model Evaluation

### The Class Imbalance Problem
This was the biggest challenge. Only 29.8% of patients are readmitted, so a model that always predicts "no readmission" would get 70% accuracy!

**Solution**: Focused on ROC-AUC instead of accuracy.

### Final Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Random Forest | 62.5% | 40.2% | 52.8% | 0.46 | **0.64** |
| Gradient Boosting | 70.5% | 53.3% | 7.1% | 0.13 | **0.65** |
| Logistic Regression | 59.4% | 38.6% | 61.2% | 0.47 | **0.64** |

### Interpretation
- **ROC-AUC 0.64-0.65** is significantly better than random (0.50)
- Reflects realistic performance on noisy healthcare data
- Missing important variables (socioeconomic status, insurance, medication adherence) limits predictive power

### Statistical Validation
- **Chi-square test**: Age group significantly associated with readmission (p < 0.0001)
- **t-test**: Length of stay differs between readmitted vs not (p < 0.0001)
- **Cross-validation**: Consistent performance across folds (no overfitting)

---

## 6. Feature Importance Analysis

### Top 10 Predictive Features

1. **Prior inpatient admissions** (15.9%)
2. Total prior admissions (12.0%)
3. Emergency admission flag (8.9%)
4. Age squared (8.2%)
5. Emergency visit ratio (5.7%)
6. Has prior admission (4.6%)
7. Age × Prior admissions interaction (4.2%)
8. Elderly flag (65+) (3.9%)
9. Composite risk score (3.8%)
10. Emergency visits count (3.6%)

**Key Insight**: Patient history (prior admissions, ED visits) is the strongest predictor. Makes clinical sense - past behavior predicts future behavior.

---

## 7. Limitations & Future Work

### Current Limitations
- Synthetic data (not real patient records)
- Missing important variables (insurance status, social support, medication adherence)
- Temporal effects not modeled (seasonality, policy changes)
- Model doesn't consider cause of readmission

### Next Steps
- **SMOTE** (Synthetic Minority Oversampling) to better handle class imbalance
- **XGBoost** with tuned hyperparameters
- **Feature selection** (LASSO, recursive feature elimination)
- **Survival analysis** (time-to-readmission modeling)
- **Calibration** (ensure predicted probabilities are accurate)

**Target**: Improve ROC-AUC to 0.75-0.80 range

---

## Technical Stack

**Data Processing**
- Python 3.8+
- pandas (data manipulation)
- NumPy (numerical operations)

**Machine Learning**
- scikit-learn (models, preprocessing, metrics)
- Cross-validation, stratification

**Database**
- SQL (data extraction, aggregation)
- Simulated multi-table joins

**Analysis**
- Jupyter Notebook (exploratory analysis)
- Statistical tests (scipy.stats)

**Version Control**
- Git/GitHub

---

## Reflections

### What I Learned

**Technical Skills**
- Large-scale data processing (1M+ records)
- Feature engineering based on domain knowledge
- Handling imbalanced datasets
- Model evaluation beyond accuracy

**Statistical Thinking**
- Importance of appropriate metrics
- Hypothesis testing for validation
- Understanding Bayes error and irreducible uncertainty

**Domain Knowledge**
- Healthcare data characteristics
- Clinical relevance vs statistical significance
- Ethical considerations in medical ML

### Challenges Overcome
- Debugging SQL queries on large datasets
- Balancing model complexity vs interpretability
- Explaining why 65% AUC is actually good for this problem

---

## Author

**Jiayi Lyu**  
Mathematics Major, Statistics Minor  
University of British Columbia

GitHub: [@lyujiayi7-boop](https://github.com/lyujiayi7-boop)

---

*Last Updated: February 2026*
