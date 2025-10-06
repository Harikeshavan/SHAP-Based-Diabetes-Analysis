import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
import shap
import lime.lime_tabular
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score, fbeta_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import mutual_info_classif


import warnings
warnings.filterwarnings("ignore")

csv_fl = 'hb1ac_trn_copy.csv'
data = pd.read_csv(csv_fl)

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, fmt=".2f")
plt.title('Pre-SMOTE Correlation Matrix')
plt.show()

train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['diabetes'])
train_processed = pd.get_dummies(train)
test_processed = pd.get_dummies(test)

train_processed = train_processed.fillna(train_processed.mean())
test_processed = test_processed.fillna(test_processed.mean())

X_train = train_processed.drop(['diabetes'], axis=1)
Y_train = train_processed['diabetes']
X_test = test_processed.drop(['diabetes'], axis=1)
Y_test = test_processed['diabetes']

smenn = SMOTEENN(random_state=42)
X_train, Y_train = smenn.fit_resample(X_train, Y_train) #type:ignore

plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(X_train).corr(), annot=True, fmt=".2f")
plt.title('Post-SMOTE Correlation Matrix')
plt.show()

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, Y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy * 100)

coefficients = model.coef_[0]
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': coefficients})
importance_df['Absolute Coefficient'] = importance_df['Coefficient'].abs()
importance_df = importance_df.sort_values(by='Absolute Coefficient', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Coefficient', y='Feature', data=importance_df, palette='viridis')
plt.title('Feature Importance based on Coefficients (Logistic Regression)')
plt.show()

explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=X_train.columns)

explanation_dmy = shap.Explainer(model,X_train)
explanation = explanation_dmy(X_train[:1000])

for i in (X_train.columns):
    shap.plots.scatter(explanation[:, i])

exp_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=[str(i) for i in Y_train.unique()],  #type:ignore
    mode='classification'
)

sample_index = 5
sample = X_test.iloc[sample_index]

explained_instance = exp_lime.explain_instance(
    data_row=sample,
    predict_fn=model.predict_proba
)

explained_instance.as_pyplot_figure()
plt.show()

mi_scores = mutual_info_classif(X_train, Y_train, random_state=42) #type:ignore
mi_df = pd.DataFrame({'Feature': X_train.columns, 'Mutual Information Score': mi_scores})
mi_df = mi_df.sort_values(by='Mutual Information Score', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Mutual Information Score', y='Feature', data=mi_df, palette='plasma')
plt.title('Feature Positive Association Approximation (Mutual Information Score)')
plt.show()

# Predict and calculate AUC
Y_pred = model.predict(X_test)
Y_pred_proba = model.predict_proba(X_test)[:, 1]
auc = metrics.roc_auc_score(Y_test, Y_pred_proba)
print(f'AUC: {auc}')

accuracy = accuracy_score(Y_test, Y_pred) * 100
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
f2 = fbeta_score(Y_test, Y_pred, beta=2)
auc = roc_auc_score(Y_test, Y_pred_proba)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"F2 Score: {f2:.2f}")
print(f"AUC: {auc:.2f}")

# Plot ROC curve
fpr, tpr, _ = metrics.roc_curve(Y_test, Y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion matrix
cm = metrics.confusion_matrix(Y_test, Y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues) #type:ignore
plt.title('Confusion Matrix')
plt.show()