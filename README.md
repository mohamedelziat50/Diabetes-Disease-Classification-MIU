# Diabetes Disease Classification MIU

Diabetes disease classification with preprocessing, PCA feature reduction, and SMOTE-based imbalance handling, along with cross-validation, hyperparameter tuning, and learning-curve analysis. Compares logistic regression, Gaussian Naive Bayes, AdaBoost, Random Forest, SVM, MLP, and XGBoost using cross-validated evaluation. (Task 2 ML)

<p align="center">
	<a href="Diabetes_Classification.ipynb"><img src="https://img.shields.io/badge/Open%20Notebook-Diabetes_Classification.ipynb-0f766e?style=for-the-badge&logo=jupyter" alt="Open Notebook" /></a>
	<a href="#project-overview"><img src="https://img.shields.io/badge/Task%202%20ML-Classification-20BEFF?style=for-the-badge" alt="Task 2 ML" /></a>
	<a href="#getting-started"><img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" /></a>
	<a href="#key-highlights"><img src="https://img.shields.io/badge/PCA-Feature%20Reduction-4b5563?style=for-the-badge" alt="PCA" /></a>
	<a href="#key-highlights"><img src="https://img.shields.io/badge/SMOTE-Imbalance%20Handling-2e7d32?style=for-the-badge" alt="SMOTE" /></a>
</p>

## Project Overview

This repository contains a full end-to-end workflow for diabetes disease classification, including data cleaning, feature engineering, model training, hyperparameter tuning, and final evaluation. The notebook evaluates several traditional machine learning models and ensemble methods, then selects the final model based on F1-score because the problem is class-imbalanced.

## Key Highlights

- Duplicate rows removed: 3,854 duplicates were detected and eliminated.
- Outliers handled with IQR-based clipping.
- SMOTE applied only to the training set to avoid data leakage.
- PCA reduced the feature space to 8 components while retaining about 97% of the variance.
- Multiple models were tuned with `GridSearchCV` and `RandomizedSearchCV`.
- Final model selected: XGBoost without PCA.

## Workflow

1. Load and inspect the diabetes dataset.
2. Clean the data by removing duplicates and handling outliers.
3. Split the data into training, validation, and test sets.
4. Apply scaling, SMOTE, and PCA where appropriate.
5. Train and tune multiple classifiers.
6. Compare models with confusion matrices, precision, recall, and F1-score.
7. Select the best-performing model for the final report.

## Models Evaluated

- Logistic Regression
- Gaussian Naive Bayes
- AdaBoost
- Random Forest
- Support Vector Machine
- MLP Classifier
- XGBoost

## Results Summary

The final comparison in the report shows that XGBoost achieved the best testing F1-score and was selected as the final model. PCA did not improve the XGBoost result in this project, so the non-PCA version was kept as the final choice.

| Model | Testing F1-score | Notes |
| --- | ---: | --- |
| XGBoost | 0.8010 | Final selected model |
| Final RBF SVM | 0.7917 | Strong accuracy, slightly lower F1 |
| Random Forest | 0.7689 | Competitive baseline |
| AdaBoost | 0.6926 | Moderate performance |
| Logistic Regression | 0.5757 | Lower recall/F1 balance |
| Gaussian Naive Bayes | 0.3117 | Lowest overall F1 |
| XGBoost + PCA | 0.6509 | PCA reduced performance |

## Why F1-Score Was Used

This dataset is imbalanced, so accuracy alone can hide poor minority-class performance. F1-score provides a better balance between precision and recall, making it a better primary metric for comparing the classifiers in this project.

## Repository Structure

```text
.
├── Diabetes_Classification.ipynb
├── README.md
├── dataset/
│   └── diabetes_prediction_dataset.csv
└── Documents/
	├── Task 2 - Diabetes Disease Classification.docx
	└── IEEE_Diabetes_Classification.pdf
```

## Getting Started

### Requirements

- Python 3.10+
- Jupyter Notebook or Google Colab
- Packages used in the notebook: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `scipy`

## Project Outcome

The project demonstrates that careful preprocessing, imbalanced-data handling, and model tuning can materially improve diabetes classification performance. Among the tested models, XGBoost without PCA provided the best overall balance of predictive performance and robustness.

## Acknowledgment

This repository was developed as part of the MIU diabetes disease classification assignment.