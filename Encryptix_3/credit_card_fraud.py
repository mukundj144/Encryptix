import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

credit=pd.read_csv(r'creditcard.csv')

credit.head(7)

credit.info()

credit.describe()

credit['Class'].value_counts()

# here log scale is used as the dataset is highly imbalanced
plt.figure(figsize=(7, 5))
sns.countplot(x='Class', data=credit)
plt.yscale('log')
plt.title('Class Distribution (Logarithmic Scale)')
plt.xlabel('Class')
plt.ylabel('Count (log scale)')
plt.show()

# Normalize the 'Amount' and 'Time' features
scaler = StandardScaler()
credit['Amount'] = scaler.fit_transform(credit[['Amount']])
credit['Time'] = scaler.fit_transform(credit[['Time']])

# Define features and target variable
X = credit.drop(columns=['Class'])
y = credit['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Initialize the Random Forest classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
import time
start_time = time.time()
rf_clf.fit(X_train_resampled, y_train_resampled)
end_time = time.time()

training_time = end_time - start_time

print(f"Time taken to train the Random Forest model: {training_time:.2f} seconds")

# Predict probabilities for the test set
y_probs = rf_clf.predict_proba(X_test)[:, 1]

# Predict classes for the test set
y_pred = rf_clf.predict(X_test)

# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print precision, recall, and F1-score
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Calculate precision and recall values for the Precision-Recall curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)

# Calculate the Area Under the Precision-Recall Curve (AUPRC)
auprc = auc(recall_curve, precision_curve)

# Print the AUPRC
print(f'Area Under the Precision-Recall Curve (AUPRC): {auprc:.4f}')

# Plot the Precision-Recall curve
plt.figure()
plt.plot(recall_curve, precision_curve, label=f'Random Forest (AUPRC = {auprc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud']))