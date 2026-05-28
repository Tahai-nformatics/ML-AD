# ML-AD  
## Machine Learning for Classification of Subjects and Disease Progression Prediction of MCI Patients to Alzheimer’s Disease

---

# Overview

This project applies **machine learning methods** to Alzheimer's Disease Neuroimaging Initiative (**ADNI**) clinical data to:

### 1. Classify Baseline Cognitive Diagnosis
- **Cognitively Normal (CN)**
- **Mild Cognitive Impairment (MCI)**
- **Alzheimer’s Disease (AD)**

### 2. Predict Disease Progression
Predict whether baseline **MCI patients convert to Alzheimer’s Disease within 3 years**.

---

# Project Goals

The project demonstrates a complete machine learning workflow in **R**, including:

- **Clinical data preprocessing**
- **Longitudinal cohort construction**
- **Feature engineering**
- **Predictive modeling**
- **Model evaluation using ROC/AUC analysis**
- **Comparison of multiple machine learning models**

---

# Dataset

This project uses data from the **Alzheimer’s Disease Neuroimaging Initiative (ADNI)** through the `ADNIMERGE2` R package.

## Key Datasets
| Dataset | Description |
|---|---|
| `ADSL` | Subject-level demographic and baseline clinical data |
| `DXSUM` | Longitudinal diagnosis and follow-up data |

---

# Clinical Features Used

The following predictors were included in the models:

- **Age**
- **Sex**
- **Education**
- **MMSE Score (`MMSCORE`)**
- **ADAS-Cog 13 (`ADASTT13`)**
- **Clinical Dementia Rating Sum of Boxes (`CDRSB`)**

---

# Project Workflow

## 1. Baseline Diagnosis Classification

A **multinomial logistic regression model** was trained to classify subjects into:

- **CN**
- **MCI**
- **AD**

### Steps
- Cleaned diagnosis categories
- Removed missing values
- Split data into training/testing sets
- Trained multinomial logistic regression model
- Evaluated predictions using:
  - **Confusion Matrix**
  - **Per-class Accuracy**
  - **ROC Curves**
  - **AUC Scores**

---

## 2. MCI Progression Prediction

A longitudinal cohort of baseline MCI patients was created to predict **conversion to AD within 3 years**.

### Cohort Construction
- Identified baseline MCI subjects
- Extracted earliest diagnosis visit date
- Calculated follow-up duration
- Determined AD conversion status
- Required **≥3 years follow-up** for confirmed non-converters

---

# Machine Learning Models

The following machine learning models were implemented and compared:

| Model | Purpose |
|---|---|
| **Logistic Regression** | Baseline binary classification |
| **Random Forest** | Non-linear ensemble learning |
| **XGBoost** | Gradient boosting classification |

---

# Model Evaluation

Models were evaluated using:

- **ROC Curves**
- **Area Under the Curve (AUC)**
- **Accuracy**
- **Sensitivity / Specificity**
- **Confusion Matrices**

The final section compares all models using a combined ROC plot to benchmark predictive performance.

---

# Technologies Used

## Programming Language
- **R**

## Libraries
```r
library(ADNIMERGE2)
library(dplyr)
library(tidyr)
library(lubridate)
library(nnet)
library(randomForest)
library(caret)
library(pROC)
library(xgboost)
```

# Future Improvements

Potential future enhancements include:

- Incorporating **genetic data (GWAS / PRS)** into prediction models
- Performing **hyperparameter tuning** for Random Forest and XGBoost
- Implementing **cross-validation** for improved model robustness
- Exploring **deep learning approaches** for disease prediction
- Expanding the pipeline to include additional clinical biomarkers
- Improving model interpretability for clinical applications

