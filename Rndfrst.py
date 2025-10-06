import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import seaborn as sns
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import (accuracy_score, roc_auc_score, roc_curve, confusion_matrix, 
                             ConfusionMatrixDisplay, precision_score, recall_score, f1_score, 
                             fbeta_score)

# Initialize SHAP visualization
shap.initjs()

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.formatter.useoffset'] = False

# Load data
csv_fl = 'hb1ac_trn_copy.csv'
data = pd.read_csv(csv_fl)

# Pre-SMOTEENN correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features pre SMOTEENN')
plt.show()

# Train-test split
train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['diabetes'])
train_processed = pd.get_dummies(train)
test_processed = pd.get_dummies(test)

train_processed = train_processed.fillna(train_processed.mean())
test_processed = test_processed.fillna(test_processed.mean())

X_train = train_processed.drop(['diabetes'], axis=1)
Y_train = train_processed['diabetes']

X_test = test_processed.drop(['diabetes'], axis=1)
Y_test = test_processed['diabetes']

# SMOTEENN resampling
smenn = SMOTEENN(random_state=42)
X_train, Y_train = smenn.fit_resample(X_train, Y_train)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Post-SMOTEENN correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(pd.DataFrame(X_train).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Features post SMOTEENN')
plt.show()

# Random Forest Classifier
random_forest = rfc(n_estimators=470)
random_forest.fit(X_train, Y_train)
random_forest_preds = random_forest.predict(X_test)
print('The accuracy of the Random Forest model is :', metrics.accuracy_score(random_forest_preds, Y_test) * 100)

Y_pred = random_forest.predict(X_test)
Y_pred_proba = random_forest.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, Y_pred_proba)

accuracy = accuracy_score(Y_test, Y_pred) * 100
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
f2 = fbeta_score(Y_test, Y_pred, beta=2)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"F2 Score: {f2:.2f}")
print(f"AUC: {auc:.2f}")

# SHAP with K-Means to reduce sample size
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)
cluster_labels = kmeans.predict(X_train)

# Selecting a representative sample from each cluster
sampled_indices = []
for cluster in np.unique(cluster_labels):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    sampled_indices.extend(np.random.choice(cluster_indices, size=min(50, len(cluster_indices)), replace=False))

X_train_sampled = X_train.iloc[sampled_indices]
Y_train_sampled = Y_train.iloc[sampled_indices]

# SHAP Kernel Explainer on sampled data
explainer = shap.KernelExplainer(random_forest.predict, X_train_sampled, check_additivity=False)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
shap.summary_plot(shap_values, X_test, plot_type='bar')

# LIME Explainer
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train_sampled), 
    feature_names=X_train_sampled.columns, 
    class_names=['No Diabetes', 'Diabetes'], 
    mode='classification'
)

sample_index = 5
sample = X_test.iloc[sample_index]
explained_instance = lime_explainer.explain_instance(
    data_row=sample.values, 
    predict_fn=random_forest.predict_proba
)
explained_instance.as_pyplot_figure()
plt.show()

# Random Forest Evaluation
Y_pred = random_forest.predict(X_test)
Y_pred_proba = random_forest.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, Y_pred_proba)

accuracy = accuracy_score(Y_test, Y_pred) * 100
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)
f2 = fbeta_score(Y_test, Y_pred, beta=2)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"F2 Score: {f2:.2f}")
print(f"AUC: {auc:.2f}")

# Plot ROC Curve
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()