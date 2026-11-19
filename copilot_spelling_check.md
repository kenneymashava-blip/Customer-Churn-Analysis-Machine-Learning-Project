**Click on the link to launch the Streamlit app 👉👉👉**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-analysis-machine-learning-project-p64ajwiykxu4q.streamlit.app/)

# Customer-Churn-Analysis-Machine-Learning-Project
The main focus of this project is to develop a machine learning model using scikit-learn to predict the customers who are likely to leave the business and deploy the model using Streamlit.

## Tech Stack

- **Python**
- **Pandas, NumPy** (data cleaning & processing)
- **Seaborn, Matplotlib, Plotly Express** (visualizations)
- **Scikit-learn** (modeling)
- **Streamlit, Joblib, Visual Studio Code** (web app development and deployment)
- **Jupyter Notebook**

# Data Cleaning
- The only data cleaning performed on the dataset was to drop one row containing missing values, at row 199295.

# Exploratory Data Analysis (EDA)

The dataset was explored using Python visualization libraries such as Seaborn, Matplotlib, and Plotly Express.

**Key visualizations included:**

- Count plot showing the overall churn rate
- Histograms showing numerical feature distributions by churn
- Count plots showing categorical feature distributions by churn
- Box-and-whisker plots of numerical features by churn
- Correlation heatmap to detect multicollinearity among numerical features
- Correlation bar plot of numerical features against churn

***

**Key findings from the categorical feature visuals:**

- The dataset consisted of 56% churners (1) and 44% non-churners (0). This class imbalance was handled during model training.
- Monthly contract customers had the highest churn rate in the contract-length category: 90% churners versus 10% non-churners.
- The female gender category consisted of 65% churners and 35% non-churners.

***

**Key findings from the numerical feature visuals:**

- Customers aged 50 to 65 had an 84% churn rate.
- Customers who made more than four support calls had an 84% churn rate.
- Total spend below $500 was associated with a 90% churn rate.
- Payment delays exceeding 20 days were associated with an 87% churn rate.
- Customers with more than 17 days since their last interaction had a 64% churn rate.

***

**Key findings from numerical feature correlations with churn:**

- No multicollinearity was detected among the numerical features.
- The features most correlated with churn were:
- Support Calls (correlation p = 0.5163)
- Total Spend (p = -0.3697)
- Payment Delay (p = 0.3298)
- The remaining numerical features had correlations below 0.2.

# Machine Learning Modeling

Three machine learning models—Logistic Regression, Random Forest Classifier, and XGBoost Classifier—were fitted using the training data. The best model was selected based on recall.

***

## Preprocessing

- The data was split into training and testing datasets and stratified by churn (the target labels).
- For the **Logistic Regression preprocessor**, **OneHotEncoder** for categorical features and a **StandardScaler** for numerical features were used to initialize the **ColumnTransformer**.
- To prevent multicollinearity, **OneHotEncoder(drop='first')** was used for the Logistic Regression preprocessor.
- The **Random Forest** and **XGBoost preprocessors** only required **OneHotEncoder(drop=None)** because tree-based models use all available predictors for maximum learning.

***

## Cross-validation and hyperparameter tuning

- Pipelines combining the preprocessors and hyperparameters for each model were initialized.
- Coupling the preprocessors with the pipelines prevented data leakage during model training and cross-validation.
- GridSearchCV was fitted with the training data to find the best model pipeline and hyperparameter combination.
- The metric maximized by GridSearchCV was accuracy.
- Class imbalance in the training dataset was handled using the `class_weight` parameter in the model pipelines.
- The best model was the XGBoost classifier pipeline.

***

## Trained model metrics

| Metric | Logistic Regression | Random Forest Classifier | XGBoost Classifier |
| :------------------ | :----------------------- | :------------------------ | :------------------- |
| Accuracy Score (1 & 0) | 0.8417 | 0.9313 | 0.9333 |
| Precision (1) | 0.8861 | 0.8970 | 0.8969 |
| Recall (1) | 0.8204 | 0.9899 | 0.9940 |
| F1 Score (1) | 0.8520 | 0.9412 | 0.9430 |
| Support Samples (1) | 84148 | 84148 | 84148 |

***

## Feature importance plot for the best model (XGBoost classifier)

<img width="1000" height="500" alt="XGBoost feature importance plot" src="https://github.com/user-attachments/assets/cb77ca1b-afa5-4614-9e82-24aafd59919b" />

***

### Key takeaways from feature importances

- The six most important features for predicting churn, in descending order, were:
  - Number of support calls
  - Monthly contract
  - Female gender
  - Total spend
  - Payment delay
  - Age
- The remaining features did not contribute significantly to churn prediction.
- Surprisingly, the male gender feature contributed zero importance to churn prediction.

***

# Streamlit Web-App Development

- The best model (XGBoost classifier pipeline) was saved using Joblib.
- The feature-importance dataframe was saved as a CSV file.
- Both resources were used in developing the Streamlit web app.
- **The Streamlit web app Python script and all related resources can be found in the `Streamlit_App` folder.**

**Click on the link to launch the Streamlit app 👉👉👉**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-analysis-machine-learning-project-p64ajwiykxu4q.streamlit.app/)

