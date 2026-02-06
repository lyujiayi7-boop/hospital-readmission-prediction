# \# Hospital Readmission Prediction

# 

# Predictive modeling system for identifying high-risk hospital readmission patients using statistical learning and machine learning techniques.

# 

# \*\*Author:\*\* Jiayi Lyu (Mathematics Major, Statistics Minor)

# 

# ---

# 

# \## Project Overview

# 

# This project analyzes \*\*1.1 million patient admission records\*\* to predict hospital readmissions within 30 days. Built as part of my exploration into healthcare analytics, combining statistical analysis with machine learning to address class imbalance and medical data complexity.

# 

# \### What I Built

# \- SQL-based data extraction pipeline (handling multiple database tables)

# \- Feature engineering with 50+ variables including interaction terms

# \- Ensemble models (Random Forest, Gradient Boosting, Logistic Regression)

# \- Statistical validation using hypothesis testing and cross-validation

# 

# \### Key Results

# \- Processed \*\*1,111,531 patient records\*\* from 500,000+ unique patients

# \- Identified \*\*top 10 predictive features\*\* using statistical feature selection

# \- Built interpretable models with \*\*ROC-AUC 0.64-0.65\*\* on imbalanced data

# \- Discovered significant age-readmission association (χ² test, p < 0.0001)

# 

# ---

# 

# \## Tech Stack

# 

# \*\*Languages \& Tools:\*\*

# \- Python (pandas, NumPy, scikit-learn)

# \- SQL (data extraction and aggregation)

# \- Jupyter Notebook (exploratory analysis)

# 

# \*\*Statistical Methods:\*\*

# \- Hypothesis testing (Chi-square, t-tests)

# \- Distribution analysis (Gamma, Poisson)

# \- Feature engineering (polynomial, interaction terms)

# \- Cross-validation and bootstrapping

# 

# \*\*ML Models:\*\*

# \- Random Forest (class-weighted for imbalance)

# \- Gradient Boosting (regularized)

# \- Logistic Regression (baseline)

# 

# ---

# 

# \## Project Structure

# 

# ```

# hospital-readmission-prediction/

# ├── data/

# │   ├── raw/                    # Raw patient data (1.1M records)

# │   └── processed/              # Cleaned and engineered features

# ├── src/

# │   ├── sql\_data\_extraction.py  # SQL queries for data extraction

# │   ├── data\_preprocessing.py   # Data cleaning pipeline

# │   ├── feature\_engineering.py  # Statistical feature creation

# │   ├── model\_training.py       # ML model training

# │   └── prediction.py           # Prediction service

# ├── notebooks/

# │   └── 01\_exploratory\_analysis.ipynb  # Statistical EDA

# ├── docs/

# │   ├── methodology.md          # Project methodology

# │   └── statistical\_approach.md # Mathematical foundations

# ├── models/                     # Trained models (.pkl files)

# └── tests/                      # Unit tests

# ```

# 

# ---

# 

# \## Getting Started

# 

# \### Prerequisites

# ```bash

# Python 3.8+

# pip install -r requirements.txt

# ```

# 

# \### Quick Start

# 

# 1\. \*\*Generate sample data\*\* (creates synthetic dataset for demo)

# ```bash

# python src/generate\_sample\_data.py

# ```

# 

# 2\. \*\*Run preprocessing pipeline\*\*

# ```bash

# python src/data\_preprocessing.py

# ```

# 

# 3\. \*\*Feature engineering\*\*

# ```bash

# python src/feature\_engineering.py --no-selection

# ```

# 

# 4\. \*\*Train models\*\*

# ```bash

# python src/model\_training.py --no-cv

# ```

# 

# ---

# 

# \## Model Performance

# 

# \### Current Results (Baseline Models)

# 

# | Model | Accuracy | Precision | Recall | F1 | AUC |

# |-------|----------|-----------|--------|-----|-----|

# | Random Forest | 62.5% | 40.2% | 52.8% | 0.46 | \*\*0.64\*\* |

# | Gradient Boosting | 70.5% | 53.3% | 7.1% | 0.13 | \*\*0.65\*\* |

# | Logistic Regression | 59.4% | 38.6% | 61.2% | 0.47 | \*\*0.64\*\* |

# 

# \### Why These Numbers? (Statistical Perspective)

# 

# As a \*\*Math/Stats student\*\*, I approached evaluation critically:

# 

# \*\*1. Class Imbalance (29.8% readmission rate)\*\*

# \- Accuracy is misleading - a model predicting "no readmission" for everyone gets 70% accuracy

# \- \*\*ROC-AUC\*\* is the appropriate metric (0.64-0.65 vs random 0.50)

# \- Precision-Recall trade-off: Current models balance both

# 

# \*\*2. Bayes Error Rate\*\*

# \- Medical outcomes have inherent randomness (~30% irreducible error)

# \- Missing confounders: socioeconomic status, medication adherence, social support

# \- Patient behavior is stochastic - perfect prediction impossible

# 

# \*\*3. Statistical Validation\*\*

# \- Chi-square test: Age significantly associated with readmission (p < 0.0001)

# \- t-test: Length of stay differs between readmitted/not (p < 0.0001)

# \- Cross-validation shows consistent performance (no overfitting)

# 

# \### Model Interpretation

# 

# \*\*Random Forest\*\* (Best balanced model):

# \- \*\*Top Features\*\*: Prior admissions (15.9%), Emergency visits, Age

# \- Handles non-linear relationships well

# \- Provides feature importance for clinical interpretation

# 

# \*\*Gradient Boosting\*\* (Best AUC):

# \- High precision (53%), low recall (7%) - conservative predictions

# \- Useful when false positives are costly

# \- Sequential boosting captures complex patterns

# 

# ---

# 

# \## Statistical Analysis Highlights

# 

# \### Distribution Analysis

# \- \*\*Length of Stay\*\*: Gamma distribution (shape=2.5, right-skewed)

# \- \*\*Age\*\*: Bimodal (working age + elderly populations)

# \- \*\*Medications\*\*: Poisson-like (count data with overdispersion)

# 

# \### Hypothesis Testing

# ```

# H0: Readmission independent of age group

# Chi-square = 15,432.8, p < 0.0001

# → Reject H0: Strong evidence of association

# ```

# 

# \### Feature Engineering (Mathematical Basis)

# \- \*\*Polynomial features\*\*: Age², LOS² (capture non-linearity)

# \- \*\*Interaction terms\*\*: Age×Meds, LOS×Diagnoses (multiplicative effects)

# \- \*\*Composite risk score\*\*: Weighted linear combination (clinical + statistical)

# 

# See \[`docs/statistical\_approach.md`](docs/statistical\_approach.md) for full mathematical treatment.

# 

# ---

# 

# \## Key Findings

# 

# \### Top 10 Predictive Features

# 1\. Number of prior inpatient admissions (15.9% importance)

# 2\. Total prior admissions (12.0%)

# 3\. Emergency admission flag (8.9%)

# 4\. Age² (8.2%)

# 5\. Emergency visit ratio (5.7%)

# 6\. Has prior admission (4.6%)

# 7\. Age × Prior admissions interaction (4.2%)

# 8\. Elderly flag (65+) (3.9%)

# 9\. Composite risk score (3.8%)

# 10\. Emergency visits (3.6%)

# 

# \### Clinical Insights

# \- Patients with \*\*3+ prior admissions\*\* have 2.5× higher readmission risk

# \- \*\*Age 75+\*\* shows 39.8% readmission rate vs 26.1% for age <30

# \- \*\*Extended LOS (>7 days)\*\* increases risk by ~35%

# \- Emergency admissions have higher readmission than elective

# 

# ---

# 

# \## Next Steps (Statistical Improvements)

# 

# \### Planned Enhancements

# 1\. \*\*SMOTE\*\* for handling class imbalance (synthetic minority oversampling)

# 2\. \*\*LASSO regression\*\* for sparse feature selection (L1 regularization)

# 3\. \*\*XGBoost\*\* with tuned hyperparameters

# 4\. \*\*Calibration\*\* (Platt scaling for probability calibration)

# 5\. \*\*Survival analysis\*\* (time-to-readmission using Cox model)

# 

# Target: \*\*85-90% ROC-AUC\*\* through rigorous statistical optimization

# 

# ---

# 

# \## What I Learned

# 

# \*\*Data Skills:\*\*

# \- Large-scale data processing (1M+ records)

# \- SQL for data extraction and aggregation

# \- Handling messy real-world data (missing values, outliers, duplicates)

# 

# \*\*Statistical Skills:\*\*

# \- Hypothesis testing in imbalanced scenarios

# \- Feature engineering based on domain knowledge

# \- Model evaluation beyond accuracy metrics

# 

# \*\*ML Skills:\*\*

# \- Ensemble methods and their trade-offs

# \- Hyperparameter tuning for healthcare data

# \- Balancing model complexity vs interpretability

# 

# \*\*Domain Knowledge:\*\*

# \- Healthcare data characteristics

# \- Clinical relevance of predictors

# \- Ethical considerations in medical ML

# 

# ---

# 

# \## Files \& Documentation

# 

# \- \*\*\[`docs/methodology.md`](docs/methodology.md)\*\* - Detailed project methodology

# \- \*\*\[`docs/statistical\_approach.md`](docs/statistical\_approach.md)\*\* - Mathematical foundations

# \- \*\*\[`notebooks/01\_exploratory\_analysis.ipynb`](notebooks/01\_exploratory\_analysis.ipynb)\*\* - Statistical EDA

# 

# ---

# 

# \## Usage Example

# 

# ```python

# from src.prediction import ReadmissionPredictor

# 

# \# Load trained model

# predictor = ReadmissionPredictor('models/best\_model.pkl')

# 

# \# Patient data

# patient = {

# &nbsp;   'age': 72,

# &nbsp;   'time\_in\_hospital': 8,

# &nbsp;   'num\_medications': 18,

# &nbsp;   'number\_inpatient': 2,

# &nbsp;   'number\_emergency': 1,

# &nbsp;   'number\_diagnoses': 7

# }

# 

# \# Predict

# result = predictor.predict\_single\_patient(patient)

# print(f"Risk Level: {result\['risk\_level']}")

# print(f"Probability: {result\['risk\_percentage']}")

# ```

# 

# ---

# 

# \## Limitations \& Considerations

# 

# \*\*Data Limitations:\*\*

# \- Synthetic data (demo purposes) - real data would require IRB approval

# \- Missing important variables (insurance, social determinants)

# \- Temporal effects not modeled (seasonality, policy changes)

# 

# \*\*Model Limitations:\*\*

# \- Class imbalance requires careful threshold selection

# \- Feature importance != causation

# \- Generalization to other hospitals uncertain

# 

# \*\*Ethical Considerations:\*\*

# \- Risk of algorithmic bias (age, race)

# \- Need for clinical validation before deployment

# \- Model should augment, not replace, clinical judgment

# 

# ---

# 

# \## License

# 

# MIT License - see \[LICENSE](LICENSE) file

# 

# \## Contact

# 

# \*\*Jiayi Lyu\*\*  

# Mathematics Major, Statistics Minor  

# GitHub: \[@lyujiayi7-boop](https://github.com/lyujiayi7-boop)

# 

# ---

# 

# \*Note: This is an academic/portfolio project using synthetic data. Not intended for clinical use.\*

