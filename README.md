# ML_Assignment
Machine Learning Assignment


# Mushroom Classification Using Machine Learning

## 1. Problem Statement

The objective is to develop a machine learning application
that classifies mushrooms into two categories:

- Edible
- Poisonous

Five classification algorithms are implemented and evaluated
using six standard machine learning metrics.

## 2. Dataset Description

Dataset: Mushroom Dataset

Source: UCI Machine Learning Repository

URL:
https://archive.ics.uci.edu/dataset/73/mushroom

Dataset characteristics:

- Total records: 8,124
- Input features: 22
- Target column: class
- Target classes: Edible and Poisonous
- Classification: Binary
- Feature type: Categorical

Missing values are handled during preprocessing.

## 3. GitHub Repository Link

[Add your GitHub repository URL]

## 4. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine

### Evaluation Metrics

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient

### Model Comparison

Actual evaluation values are generated in metrics_summary.csv.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD | TBD | TBD |
| Naive Bayes | TBD | TBD | TBD | TBD | TBD | TBD |
| Support Vector Machine | TBD | TBD | TBD | TBD | TBD | TBD |

### Model Performance Observations

Observations will be completed using actual test results.

- Logistic Regression: Evaluates the linear classification
  performance on encoded categorical attributes.
- Decision Tree: Evaluates feature-based decision rules.
- KNN: Evaluates classification based on neighboring samples.
- Naive Bayes: Evaluates probabilistic classification.
- Support Vector Machine: Evaluates nonlinear class separation.

### Overall Winner

To be identified using actual evaluation results,
considering accuracy, AUC and MCC.

## 5. Streamlit Application

The application includes:

- CSV upload
- Model selection
- Six evaluation metrics
- Confusion matrix
- Classification report
- Sample prediction results
- Prediction CSV download

The application loads saved models and does not
retrain on uploaded test data.

## 6. Project Execution

Install dependencies:

pip install -r requirements.txt

Train the models:

python train_models.py

Run the application:

streamlit run app.py

## 7. Deployment

GitHub Repository:
[Add actual URL]

Live Streamlit Application:
[Add actual URL]

## 8. Conclusion

This project compares five classification algorithms
using the UCI Mushroom dataset.

The evaluation results help identify the best-performing
classification model for this dataset.

This project is an academic demonstration only.
Predictions must not be used to determine whether
real mushrooms are safe to consume.
