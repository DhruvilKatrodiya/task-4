# 🔍 Logistic Regression Binary Classifier

This project builds a binary classification model using **Logistic Regression** to classify breast cancer tumors as **Malignant (M)** or **Benign (B)** using the **Wisconsin Breast Cancer Dataset**.

---

## 🎯 Objective

To implement a logistic regression model that:
- Predicts binary outcomes (`diagnosis`: M = 1, B = 0)
- Evaluates performance using accuracy, precision, recall, ROC-AUC
- Plots ROC curve and allows threshold tuning
- Explains the sigmoid function used for probability prediction

---

## 📁 Dataset

- **File:** `data.csv`
- **Target column:** `diagnosis` (converted: M → 1, B → 0)
- **Features:** 30 numerical columns describing tumor characteristics

---

## ⚙️ Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
```

---

## 🚀 Full Code with Description

```python
# Step 1: Load the dataset
df = pd.read_csv("data.csv")

# Step 2: Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Step 3: Convert the diagnosis column to numeric
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # M = Malignant, B = Benign

# Step 4: Separate features and target
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the feature values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Step 9: Evaluation Metrics
print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

print("📈 ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Step 10: Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Step 11: Custom Threshold Evaluation
threshold = 0.3
y_custom = (y_proba >= threshold).astype(int)
print(f"\n🛠 Confusion Matrix (threshold = {threshold}):")
print(confusion_matrix(y_test, y_custom))

# Step 12: Sigmoid Function Explanation
print("""
🧠 Sigmoid Function:
---------------------
    sigmoid(z) = 1 / (1 + exp(-z))

The logistic regression model uses this function to map real-valued inputs (z) 
into a range between 0 and 1, interpreted as a probability.
""")
```

---

## 📈 Output Example

```
Confusion Matrix:
[[70  2]
 [ 4 38]]

Classification Report:
              precision    recall  f1-score   support
           0       0.95      0.97      0.96        72
           1       0.95      0.90      0.92        42

ROC-AUC Score: 0.99



## 📊 Visualization

The ROC Curve visually represents the trade-off between true positive rate (TPR) and false positive rate (FPR).


## 🧪 Sigmoid Formula

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Used in logistic regression to map inputs to a 0–1 probability scale.


## 📘 License

For academic and research use.


