# SHAP-Based Diabetes Analysis

This repository implements an **explainable machine learning pipeline** for **diabetes prediction** using **Linear Regression**, **Random Forest**, and **XGBoost**, complemented with **SHAP** and **LIME** for interpretability. **SMOTE-ENN** is used to balance classes, improving model generalization. The focus is on combining predictive accuracy with transparency for interpretable healthcare AI.

---

## Objective

To build reliable diabetes classification models, analyze their decision-making process using **SHAP** and **LIME**, and understand which features most influence predictions across linear and ensemble models.

---

## Workflow Overview

1. **Data Preprocessing:**  
   - Dataset loaded from `hb1ac_trn copy.csv`.  
   - Categorical encoding using `pd.get_dummies()`.  
   - Missing values imputed with column-wise means.  
   - Stratified train-test split to preserve class distribution.  

2. **Data Balancing (SMOTE-ENN):**  
   - Corrected class imbalance while reducing noise.  
   - Pre- and post-balancing feature correlations visualized via heatmaps.

3. **Model Training:**  
   - **Linear Regression** for baseline performance.  
   - **Random Forest** to capture non-linear relationships.  
   - **XGBoost** for optimized ensemble performance.  
   - Evaluated using **accuracy**, **precision**, **recall**, **F1**, **F2**, and **AUC**.

4. **Model Explainability:**  
   - **SHAP:** Global and local feature impact visualization for all models.  
   - **LIME:** Instance-level explanations for sample predictions.  
   - **Mutual Information:** Non-linear feature relevance estimation.

5. **Performance Visualization:**  
   - ROC curves, confusion matrices, and feature importance plots generated for all models.

---

## Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | F2 Score |
|--------------------|---------|-----------|--------|----------|----------|
| **XGBoost**        | 95.50%  | 0.64      | 0.84   | 0.72     | 0.79     |
| **Random Forest**   | 93.44%  | 0.58      | 0.84   | 0.68     | 0.77     |
| **Linear Regression** | 87.15% | 0.39      | 0.89   | 0.54     | 0.71     |

---

## Key Insights

- **SMOTE-ENN** improved class balance and reduced bias across models.  
- **XGBoost** achieved the highest overall performance.  
- **Linear Regression** showed high recall but lower precision, useful for detecting positive cases.  
- **SHAP and LIME** provide interpretable insights for all models, identifying features that drive predictions and enhancing transparency.
