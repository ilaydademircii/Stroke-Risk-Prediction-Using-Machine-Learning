# Stroke Risk Prediction Using Machine Learning

## About the Project

This project aims to predict stroke risk using logistic regression on a medical dataset containing patient features such as age, gender, hypertension, heart disease, glucose levels, BMI, and smoking status. The data is preprocessed and modeled to classify individuals as at risk or not at risk of stroke, facilitating early intervention.

The dataset used is publicly available on Kaggle:
[Stroke Risk Prediction Dataset v2](https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset-v2/data)

## Technologies Used

* Python
* Pandas, NumPy (data handling)
* Matplotlib, Seaborn (visualization)
* Scikit-learn (modeling, evaluation)
* Warnings module for clean output

 ### 📄 Project Reports

- [Exploratory Data Analysis (EDA) Report](https://github.com/ilaydademircii/Stroke-Risk-Prediction-Using-Machine-Learning/blob/main/Exploratory%20Data%20Analysis%20(EDA).pdf)  
  A detailed report on exploratory data analysis, visualization, and evaluation of key features related to stroke risk.

- [Comprehensive Project Report](https://github.com/ilaydademircii/Stroke-Risk-Prediction-Using-Machine-Learning/blob/main/Proje%20Raporu.pdf)  
  A thorough report covering the methodology, implementation steps, results, and potential future improvements of the project.



📊 ## Exploratory Data Analysis (EDA)

A comprehensive exploratory data analysis is performed, including approximately 12 different visualizations to better understand the data distribution and relationships, such as:

* Feature distributions (age, glucose level, BMI, etc.)
* Relationships between categorical variables (gender, smoking status) and stroke risk
* Correlation heatmaps to identify multicollinearity
* Count plots and bar charts of risk factors
* Missing data patterns visualization

These insights guide feature selection and preprocessing decisions.

## Data Preprocessing

* Loaded dataset with pandas, checked for missing values and data types
* Converted categorical variables (e.g., gender) into dummy/indicator variables
* Dropped `stroke_risk_percentage` to avoid data leakage

## Model Training and Evaluation

* Split data into training (80%) and testing (20%) sets
* Trained a **Logistic Regression** model with max iterations set to 1000
* Generated predictions on the test set
* Evaluated model using accuracy, confusion matrix, classification report
* Calculated detailed metrics: sensitivity (recall), specificity, precision, and F1-score
* Visualized confusion matrices with seaborn heatmaps and sklearn’s ConfusionMatrixDisplay
* Conducted permutation feature importance to identify key predictors

## Usage

1. Load and preprocess the data:

```python
import pandas as pd
df = pd.read_csv('path_to_dataset.csv', encoding='ascii')
# Preprocess categorical variables and handle missing values as needed
```

2. Define features and target, drop leakage columns:

```python
X = df.drop(columns=['at_risk', 'stroke_risk_percentage'])
y = df['at_risk']
```

3. Split data and train logistic regression:

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

4. Predict and evaluate:

```python
from sklearn.metrics import accuracy_score, classification_report

predictions = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, predictions):.4f}')
print(classification_report(y_test, predictions))
```

5. Visualize confusion matrix and feature importance as shown in the main code.

## Model Performance

* Accuracy: \~98%
* Sensitivity (Recall), Specificity, Precision, and F1-Score all above 0.97
* Confusion matrices demonstrate effective classification of stroke risk
* Permutation importance reveals the most influential features on the prediction

## Notes

* The `stroke_risk_percentage` feature is excluded to prevent data leakage
* Warnings are suppressed for cleaner output
* Matplotlib’s 'Agg' backend is used for non-interactive environments

## Contributing

Contributions and improvements are welcome! Feel free to fork, enhance preprocessing, experiment with other classifiers, or add visualization improvements.

