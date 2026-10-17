# Loan Approval Prediction

A machine learning project that explores applicant data and predicts loan approval outcomes using a Support Vector Machine (SVM) classifier.

## Project Workflow

1. **Data Exploration** — inspect feature distributions, missing values, and approval patterns.
2. **Data Preparation and Feature Engineering** — remove the applicant identifier, encode the target (`Y` = 1, `N` = 0), create `Total_Income`, and save the prepared dataset.
3. **Model Training** — split the data into stratified training and test sets, then tune an SVM model using five-fold cross-validation. Imputation, numerical scaling, and categorical encoding are performed inside the scikit-learn pipeline to help prevent data leakage.
4. **Model Evaluation** — evaluate the selected model on the held-out test set using classification metrics and visualizations.

## Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook

## Getting Started

1. Clone the repository using `git clone <repository-url>`.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter`.
3. Place `Loan-Approval-Prediction.csv` in `data/raw/`.
4. Run the notebooks in numerical order.

## Disclaimer

This project is for educational purposes and is not intended for real-world lending decisions.