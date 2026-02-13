# Statistical Approach to Hospital Readmission Prediction

**Author:** Jiayi Lyu  
Mathematics Major, Statistics Minor  
University of British Columbia

---

## Overview

This document outlines the statistical and mathematical foundations of this project. Rather than treating machine learning as a black box, I focused on understanding the probability theory, statistical inference, and mathematical modeling underlying the predictions.

---

## 1. Problem Formulation

### Binary Classification Problem

Given patient features **x** ∈ ℝⁿ, predict the probability of readmission:

```
P(Y = 1 | X = x) where Y ∈ {0, 1}
```

**Goal**: Build a function f: ℝⁿ → [0,1] that estimates this probability.

### The Class Imbalance Challenge

Our dataset has a 70:30 imbalance (70% not readmitted, 30% readmitted). This means:
- A naive model predicting "no readmission" for everyone gets 70% accuracy
- **Accuracy is a misleading metric** - we need probability-based evaluation

### Objective Function

To handle imbalance, we use weighted cross-entropy loss:

```
L(θ) = -∑[w₁·y·log(ŷ) + w₀·(1-y)·log(1-ŷ)]
```

where w₁ and w₀ are class weights (higher weight for minority class).

---

## 2. Statistical Analysis

### Distribution Analysis

Before modeling, I analyzed the distribution of key variables:

**Length of Stay (LOS)**
- Distribution: Right-skewed, approximately Gamma(shape=2.5, scale=1.5)
- Mean: 4.2 days, Median: 3.0 days
- Interpretation: Most patients stay 2-4 days, with long tail of extended stays

**Age**
- Distribution: Bimodal mixture
  - Mode 1: ~45 years (working age admissions)
  - Mode 2: ~75 years (elderly population)
- This informed the "elderly flag" feature (age ≥ 65)

**Number of Medications**
- Distribution: Right-skewed, overdispersed (variance > mean)
- Approximated by Gamma distribution
- Motivated polynomial feature (num_medications²)

### Hypothesis Testing

**Test 1: Chi-Square Test of Independence**

Question: Is age group associated with readmission?

```
H₀: Readmission is independent of age group
H₁: Association exists
```

Results:
- χ² statistic: 15,432.8
- Degrees of freedom: 5
- p-value: < 0.0001

**Conclusion**: Strong evidence that age and readmission are associated. Reject H₀.

**Test 2: Two-Sample t-Test**

Question: Do readmitted patients have longer hospital stays?

```
H₀: μ_readmitted = μ_not_readmitted
H₁: μ_readmitted > μ_not_readmitted
```

Results:
- t-statistic: 24.6
- p-value: < 0.0001
- Mean LOS (readmitted): 5.1 days
- Mean LOS (not readmitted): 3.8 days

**Conclusion**: Readmitted patients have significantly longer stays (p < 0.0001).

---

## 3. Feature Engineering (Mathematical Foundations)

### Polynomial Features

**Rationale**: Capture non-linear relationships

Created:
- Age² (elderly patients may have exponentially higher risk)
- LOS² (very long stays suggest complexity)

**Mathematical basis**: Taylor series expansion
```
f(x) ≈ β₀ + β₁x + β₂x² + ...
```

### Interaction Terms

**Rationale**: Capture multiplicative effects

Examples:
- **Age × Medications**: Polypharmacy risk increases with age
- **LOS × Diagnoses**: More diagnoses + longer stay = higher complexity
- **Prior_admissions × Emergency_visits**: Chronic high-utilizer pattern

**Statistical model**:
```
E[Y] = β₀ + β₁X₁ + β₂X₂ + β₃(X₁·X₂)
```

The interaction term β₃ captures how the effect of X₁ changes depending on X₂.

### Composite Risk Score

Built a weighted linear combination based on clinical literature:

```
Risk Score = 0.20 × (age/100) 
           + 0.30 × (prior_admissions/max_admissions)
           + 0.20 × (LOS/max_LOS)
           + 0.15 × (num_diagnoses/max_diagnoses)
           + 0.15 × (num_medications/max_meds)
```

Weights chosen based on:
- Clinical importance from literature
- Correlation analysis with outcome
- Normalized to [0,1] scale

---

## 4. Model Selection (Statistical Learning Theory)

### Random Forest

**Why this model?**

1. **Non-parametric**: No assumption of linear separability
2. **Ensemble method**: Reduces variance through bootstrap aggregating
3. **Feature importance**: Provides interpretable rankings via Gini impurity
4. **Handles imbalance**: Built-in class weighting

**Mathematical foundation**:
```
f_RF(x) = (1/B) ∑ᵢ₌₁ᴮ T_b(x)
```
where T_b are decision trees trained on bootstrap samples.

**Variance reduction**:
```
Var(average of B trees) ≈ σ²/B
```

### Gradient Boosting

**Why this model?**

1. **Additive model**: Sequentially corrects errors
2. **Gradient descent**: Minimizes loss function iteratively
3. **Regularization**: Learning rate and max depth prevent overfitting

**Mathematical foundation**:
```
f_m(x) = f_{m-1}(x) + η · h_m(x)
```
where h_m fits the negative gradient of the loss.

### Logistic Regression (Baseline)

Simple linear model for comparison:
```
P(Y=1|x) = 1 / (1 + e^(-w·x))
```

Helps understand whether model complexity is necessary.

---

## 5. Evaluation Metrics (Statistical Interpretation)

### ROC-AUC (Primary Metric)

**Definition**: Area Under the Receiver Operating Characteristic curve

**Statistical interpretation**: 
- Probability that the model ranks a random positive example higher than a random negative example
- Equivalent to the Wilcoxon-Mann-Whitney U statistic

**Why AUC for imbalanced data?**
- Threshold-independent (evaluates all possible classification thresholds)
- Robust to class imbalance
- AUC = 0.5 is random guessing, AUC = 1.0 is perfect

**Our results**: 
- Random Forest: AUC = 0.64
- Gradient Boosting: AUC = 0.65
- Logistic Regression: AUC = 0.64

**Interpretation**: Models are better than random (0.5) but far from perfect (1.0).

### Precision-Recall Trade-off

Due to 70:30 class imbalance:

**Precision** = TP/(TP+FP)
- "Of patients we flag as high-risk, what % actually get readmitted?"
- Important when intervention resources are limited

**Recall** = TP/(TP+FN)
- "Of patients who get readmitted, what % did we identify?"
- Important to minimize missed high-risk patients

**Our approach**: Balanced F1-score (harmonic mean of precision and recall)

### Confusion Matrix Analysis

```
                 Predicted
               No      Yes
Actual No    104K     52K    → Specificity: 66.7%
Actual Yes    31K     35K    → Sensitivity: 52.8%
```

**Statistical interpretation**:
- **Type I Error (False Positive)**: 33.3%
  - Flag patient as high-risk when they're not
  - Cost: Unnecessary interventions
  
- **Type II Error (False Negative)**: 47.2%
  - Miss a high-risk patient
  - Cost: Preventable readmission (more serious)

---

## 6. Model Performance Analysis

### Current Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|-----|-----|
| Random Forest | 62.5% | 40.2% | 52.8% | 0.46 | **0.64** |
| Gradient Boosting | 70.5% | 53.3% | 7.1% | 0.13 | **0.65** |
| Logistic Regression | 59.4% | 38.6% | 61.2% | 0.47 | **0.64** |

### Why These Results? (Statistical Perspective)

**1. Bayes Error Rate**

Medical outcomes have **inherent randomness**. Even with perfect information, prediction is limited by:
- Patient non-compliance with medications
- Unpredictable external factors (accidents, infections)
- Stochastic biological processes

Estimate: **Bayes error ≈ 20-30%** for this problem

**2. Missing Confounders**

Important unmeasured variables:
- Socioeconomic status
- Insurance coverage
- Social support network
- Mental health
- Medication adherence

These create **unobserved heterogeneity**, limiting prediction accuracy.

**3. Class Imbalance**

70:30 ratio means:
- Majority class baseline: 70% accuracy
- Our models: 60-70% accuracy BUT much better AUC (0.64-0.65 vs 0.5)

**Accuracy is misleading** - AUC is the appropriate metric.

**4. Sample Size vs Complexity**

- 1.1M samples, 53 features
- Random Forest: 300 trees, max depth 20
- Risk of overfitting to noise

**Conclusion**: 0.64-0.65 AUC is **realistic for healthcare data** with inherent uncertainty.

---

## 7. Statistical Validation

### Cross-Validation

Used **5-fold stratified cross-validation**:
- Splits preserve class distribution (70:30) in each fold
- Provides unbiased estimate of generalization error
- Standard error: σ/√5 for confidence intervals

**Results**: Consistent performance across folds (AUC std dev < 0.02)
→ No evidence of overfitting

### Feature Importance Validation

Top features align with clinical knowledge:
1. Prior admissions (strongest predictor)
2. Emergency visits
3. Age
4. Length of stay
5. Number of medications

**Statistical test**: Permutation importance (randomize feature, measure AUC drop)
→ All top 10 features are statistically significant (p < 0.01)

---

## 8. Future Statistical Improvements

### 1. Advanced Resampling

**SMOTE** (Synthetic Minority Over-sampling Technique)
- Generate synthetic examples via k-nearest neighbors interpolation
- Mathematically: x_new = x_i + λ(x_neighbor - x_i), λ ∈ [0,1]

**Expected improvement**: Better recall (reduce Type II error)

### 2. Feature Selection

**LASSO Regression** (L1 regularization)
- Penalty term: λ∑|β_j| forces some coefficients to exactly zero
- Automatic feature selection
- Reduces overfitting

**Mutual Information**
- I(X;Y) = ∑∑ p(x,y) log[p(x,y)/(p(x)p(y))]
- Removes redundant features

### 3. Model Calibration

**Platt Scaling**: Fit sigmoid to map predictions to true probabilities
```
P_calibrated = 1 / (1 + e^(A·f(x) + B))
```

**Isotonic Regression**: Non-parametric calibration (monotonic transformation)

### 4. Survival Analysis

Instead of binary prediction, model **time-to-readmission**:

**Cox Proportional Hazards Model**:
```
h(t|x) = h₀(t) · e^(β·x)
```

Allows for censored data and time-varying covariates.

---

## 9. Mathematical Rigor

### Statistical Tests for Model Comparison

**McNemar's Test** (paired predictions):
- Compare whether two models make different errors
- χ² = (b-c)²/(b+c) where b, c are discordant pairs

**DeLong's Test** (compare ROC curves):
- Tests whether AUC difference is statistically significant
- Accounts for correlation (same test set)

### Confidence Intervals

Bootstrap confidence intervals for AUC:
- Resample dataset 1000 times
- Calculate AUC for each
- 95% CI: [2.5th percentile, 97.5th percentile]

**Result**: Random Forest AUC 95% CI = [0.62, 0.66]

---

## 10. Key Takeaways

### Statistical Concepts Applied

 **Probability Theory**
- Class imbalance handling
- Risk score construction
- Probabilistic predictions

 **Statistical Inference**
- Hypothesis testing (chi-square, t-tests)
- Distribution analysis (Gamma, mixture models)
- Confidence intervals

 **Mathematical Modeling**
- Feature engineering (polynomial, interactions)
- Loss functions (weighted cross-entropy)
- Ensemble methods (variance reduction)

 **Computational Statistics**
- Large-scale data processing (1.1M records)
- Cross-validation
- Bootstrap resampling

### Why This Approach Matters

As a **Math/Stats student**, I approached this problem by:
1. Understanding the **mathematical foundations** of ML algorithms
2. Using **statistical tests** to validate findings
3. Interpreting results through **probability theory**
4. Recognizing **inherent uncertainty** (Bayes error)

This is different from a pure CS approach that might focus only on maximizing accuracy metrics without understanding the statistical theory.

---

## References

**Textbooks**:
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*
- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*

**Papers**:
- Chawla, N. V., et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique"
- DeLong, E. R., et al. (1988). "Comparing the Areas under Two or More Correlated ROC Curves"

---

*Last Updated: February 2026*
