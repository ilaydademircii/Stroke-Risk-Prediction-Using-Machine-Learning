# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Import necessary libraries
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')  # use Agg backend for matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')  # in case only plt is imported

import seaborn as sns
sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay

# Just a reminder: import warnings first, as data issues can occasionally get dramatic
# Load the dataset
data_path = 'C:/Users/zehra/Downloads/archive (3)/stroke_risk_dataset_v2.csv'
df = pd.read_csv(data_path, encoding='ascii', delimiter=',')

# Display the first few rows
print('Dataset shape:', df.shape)
print(df.head())

# Display data types to infer any possible corrections
print('\nData Types:')
print(df.dtypes)

# Check for missing values
print('Missing values per column:')
print(df.isnull().sum())

# Convert 'gender' to a categorical type
df['gender'] = df['gender'].astype('category')

# One-hot encode categorical variables (here, only gender is categorical)
df = pd.get_dummies(df, columns=['gender'], drop_first=True)

# Quick overview after preprocessing
print('\nData after preprocessing:')
print(df.head())

# Note: When handling similar datasets, ensuring categorical variables are encoded properly avoids model training issues (a common stumbling block for many practitioners).

# Define features and target
# We drop 'stroke_risk_percentage' because it is likely a direct computation of the risk and may leak information
X = df.drop(columns=['at_risk', 'stroke_risk_percentage'])
y = df['at_risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model performance
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy Score: {accuracy:.4f}')
# Display confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# Confusion Matrix ve Detaylı Metrikler
cm = confusion_matrix(y_test, predictions)

# Confusion Matrix görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Karışıklık Matrisi (Confusion Matrix)')
plt.xlabel('Tahmin Edilen Değerler')
plt.ylabel('Gerçek Değerler')

# Metrikleri hesaplama
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)  # True Positive Rate (Recall)
specificity = tn / (tn + fp)  # True Negative Rate
precision = tp / (tp + fp)  # Positive Predictive Value
f1 = 2 * (precision * sensitivity) / (precision + sensitivity)

# Metrikleri yazdırma
print("\nDetaylı Metrikler:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print(f"\nSensitivity (TPR): {sensitivity:.4f}")
print(f"Specificity (TNR): {specificity:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1 Score: {f1:.4f}")

plt.tight_layout()
plt.show()

# Confusion Matrix (Test Seti)
# Test confusion matrix hesaplama
test_cm = confusion_matrix(y_test, predictions)
print("\nTest Confusion Matrix:")
print(test_cm)

# Confusion Matrix görselleştirme
plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay(confusion_matrix=test_cm,
                      display_labels=['Hayır', 'Evet']).plot(cmap='Blues')
plt.title("Test Confusion Matrix")
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Metrikleri hesaplama
tn, fp, fn, tp = test_cm.ravel()

# Özellik Önem Analizi
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

# Özellik önemlerini sırala
importance_df = pd.DataFrame({
    'Özellik': X.columns,
    'Önem Skoru': result.importances_mean,
    'Standart Sapma': result.importances_std
})
importance_df = importance_df.sort_values('Önem Skoru', ascending=True)

# Görselleştirme
plt.figure(figsize=(12, 8))
colors = sns.color_palette("husl", len(importance_df))
bars = plt.barh(range(len(importance_df)), importance_df['Önem Skoru'],
                xerr=importance_df['Standart Sapma'],
                color=colors)

# Görsel düzenlemeler
plt.yticks(range(len(importance_df)), importance_df['Özellik'])
plt.xlabel('Özellik Önem Skoru')
plt.title('Özelliklerin Model Tahminindeki Önemi')

# Değerleri çubukların üzerine yazdır
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2,
             f'{width:.4f}',
             va='center', ha='left', fontsize=10)

plt.tight_layout()
plt.show()

# Özellik önem skorlarını yazdır
print("\nÖzellik Önem Sıralaması:")
for idx, row in importance_df.iterrows():
    print(f"{row['Özellik']}: {row['Önem Skoru']:.4f} (±{row['Standart Sapma']:.4f})")

# A brief classification report for those who enjoy details
print('Classification Report:')
print(classification_report(y_test, predictions))

# Note: In predictive modeling, it is paramount to avoid data leakage. Dropping stroke_risk_percentage as a feature prevents the model from learning trivial mappings.
