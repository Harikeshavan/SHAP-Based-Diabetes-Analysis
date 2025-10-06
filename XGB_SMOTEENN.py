import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score, fbeta_score
from sklearn.feature_selection import mutual_info_classif


import warnings
warnings.filterwarnings("ignore")
plt.rcParams['axes.formatter.useoffset'] = False
shap.initjs()

csv_fl = 'hb1ac_trn_copy.csv'
data = pd.read_csv(csv_fl)

train, test = train_test_split(data, test_size=0.3, random_state=0, stratify=data['diabetes'])
train_processed = pd.get_dummies(train)
test_processed = pd.get_dummies(test)

train_processed = train_processed.fillna(train_processed.mean())
test_processed = test_processed.fillna(test_processed.mean())

X_train = train_processed.drop(['diabetes'], axis=1)
Y_train = train_processed['diabetes']

X_test = test_processed.drop(['diabetes'], axis=1)
Y_test = test_processed['diabetes']

plt.figure(figsize=(12, 10))
corr_matrix = pd.DataFrame(X_train).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=X_train.columns, yticklabels=X_train.columns, annot_kws={"size": 17}) #type:ignore
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.title('Correlation Matrix of Features pre SMOTEENN', fontsize=14)
plt.show()


smenn = SMOTEENN(random_state=42)
X_train, Y_train = smenn.fit_resample(X_train, Y_train) #type:ignore

plt.figure(figsize=(12, 10))
corr_matrix = pd.DataFrame(X_train).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", xticklabels=X_train.columns, yticklabels=X_train.columns, annot_kws={"size": 17}) #type:ignore
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.title('Correlation Matrix of Features post SMOTEENN', fontsize=14)
plt.show()

xgb_classifier = xgb.XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1, learning_rate=0.5, random_state=42)
xgb_classifier.fit(X_train, Y_train)

result = xgb_classifier.predict(X_test)
accuracy = accuracy_score(Y_test, result)
print(f"Accuracy: {accuracy * 100:.2f}%")

exp = shap.TreeExplainer(xgb_classifier)
shap_values = exp.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

explanation_dmy = shap.Explainer(xgb_classifier,X_train)
explanation = explanation_dmy(X_train[:1000])

for i in (X_train.columns):
    shap.plots.scatter(explanation[:, i])

xgb.plot_importance(xgb_classifier, importance_type="gain")
plt.title('xgboost.plot_importance(model, importance_type="gain")')
plt.show()
xgb.plot_importance(xgb_classifier, importance_type="gain")
plt.title('xgboost.plot_importance(model, importance_type="cover")')
plt.show()

exp_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train), 
    feature_names=X_train.columns,   
    class_names=['No Diabetes', 'Diabetes'],  
    mode='classification'
)

sample_index = 5 
sample = X_test.iloc[sample_index]


explained_instance = exp_lime.explain_instance(
    data_row=sample.values,  
    predict_fn=xgb_classifier.predict_proba  
)

explained_instance.as_pyplot_figure()
plt.show()

mi_scores = mutual_info_classif(X_train, Y_train, random_state=42) #type:ignore
mi_df = pd.DataFrame({'Feature': X_train.columns, 'Mutual Information Score': mi_scores})
mi_df = mi_df.sort_values(by='Mutual Information Score', ascending=False)

# Predict and calculate AUC
Y_pred = xgb_classifier.predict(X_test)
Y_pred_proba = xgb_classifier.predict_proba(X_test)[:, 1]
auc = roc_auc_score(Y_test, Y_pred_proba)
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
fpr, tpr, _ = roc_curve(Y_test, Y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
labels = ['No Diabetes', 'Diabetes']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(6, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d') #type:ignore
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('Confusion Matrix', fontsize=14)
plt.show()
