**Click the link to launch the Streamlit app 👉👉👉**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-analysis-machine-learning-project-p64ajwiykxu4q.streamlit.app/)

# Customer Churn Analysis Machine Learning Project

## Project Overview

The main focus of this project is to develop a machine learning model using Scikit-learn to predict customers who are likely to leave the business and deploy the model using Streamlit. The project explores customer behavior, identifies factors associated with churn, and provides an interactive web application for making predictions.

## About the Dataset used

Dataset used: **customer_churn_dataset-testing-master.csv** and **customer_churn_dataset-training-master.csv**, downloaded from Kaggle [here](https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset)

These datasets contains 12 feature columns. In detail, these are:

- **CustomerID:** A unique identifier for each customer
- **Age:** The age of the customer
- **Gender:** Gender of the customer
- **Tenure:** Duration in months for which a customer has been using the company's products or services
- **Usage Frequency:** Number of times that the customer has used the company’s services in the last month
- **Support Calls:** Number of calls that the customer has made to the customer support in the last month
- **Payment Delay:** Number of days that the customer has delayed their payment in the last month
- **Subscription Type:** Type of subscription the customer has choosen
- **Contract Length:** Duration of the contract that the customer has signed with the company
- **Total Spend:** Total amount of money the customer has spent on the company's products or services
- **Last Interaction:** Number of days since the last interaction that the customer had with the company
- **Churn:** Binary label indicating whether a customer has churned (1) or not (0)

## Tech Stack

- **Python**
- **Pandas, NumPy** — data cleaning and processing
- **Seaborn, Matplotlib, Plotly Express** — visualizations
- **Scikit-learn** — modeling
- **Streamlit, Joblib, Visual Studio Code** — web app development and deployment
- **Jupyter Notebook**

# Data Cleaning

- The only data-cleaning performed on the dataset was the removal of one row containing missing values at row 199295.

# Exploratory Data Analysis (EDA)

The dataset was explored using Python visualization libraries, including Seaborn, Matplotlib, and Plotly Express.

**Key visualizations included:**

- Count plot showing the overall churn rate
- Histograms showing the distributions of numerical features by churn status
- Count plots showing the distributions of categorical features by churn status
- Box-and-whisker plots showing the distributions of numerical features by churn status
- Correlation heatmap for detecting multicollinearity among numerical features
- Correlation bar plot showing numerical features associated with churn

***

**Key findings from the categorical feature visualizations:**

- The dataset consisted of 56% churners (`1`) and 44% non-churners (`0`). The class imbalance was addressed during model training.
- Monthly contract customers contributed 22% of the business: 20% were churners and 2% were non-churners.
- Both annual and quarterly contract customers contributed 39% each: 18% were churners and 21% were non-churners.
- Female customers contributed approximately 45% of the business: 29% were churners and 16% were non-churners.
- Male customers contributed approximately 56% of the business: 27% were churners and 29% were non-churners.
- The churn percentage appeared to be approximately equivalent across the subscription types.
- Basic and Standard subscriptions had 19% churn, while Premium subscriptions had 18%.
- Standard and Premium subscriptions had 15% non-churn customers, while Basic subscriptions had 14%.

***

**Key findings from the numerical feature visualizations:**

- Customers over 50 years of age contributed 23% of the business: 19% ended up churning and 4% remained customers.
- Customers who made four or more support calls per month contributed 45% of the business: 38% ended up churning and 7% remained customers.
- Customers with total spending of less than $500 contributed 29% of the business: 26% ended up churning and 3% remained customers.
- Customers who delayed payments by 20 or more days contributed 26% of the business: 23% were churners and 3% were non-churners.
- Customers who did not interact with the business for 17 or more days per month contributed 38% of the business: 24% ended up churning and 14% remained customers.

***

**Key findings regarding numerical feature correlations with churn:**

- No multicollinearity was detected among the numerical features.
- The features most correlated with churn were:
  - Support calls: correlation \(p = 0.5163\)
  - Total spend: correlation \(p = -0.3697\)
  - Payment delay: correlation \(p = 0.3298\)
- The remaining numerical features had correlations lower than 0.2.

# Machine Learning Modeling

Three machine learning models—Logistic Regression, Random Forest Classifier, and XGBoost Classifier—were fitted using the training data. The best model was selected based on the highest recall score.

***

## Preprocessing

- The data was split into training and testing datasets and stratified by churn, the target variable.
- The **Logistic Regression preprocessor** used a `OneHotEncoder` for categorical features and a standard scaler for numerical features within a `ColumnTransformer`.
- To prevent multicollinearity, `OneHotEncoder(drop='first')` was used for the Logistic Regression preprocessor.
- The **Random Forest and XGBoost preprocessors** used `OneHotEncoder(drop=None)` because all predictive features are required for maximum learning in tree-based models.

***

## Cross-Validation and Hyperparameter Tuning

- Pipelines were created to combine the preprocessors and models with their hyperparameters.
- Coupling the preprocessors with the pipelines prevented data leakage during model training and cross-validation.
- `GridSearchCV` was fitted with the training data to identify the pipeline and hyperparameter combination with the best performance.
- The metric used to evaluate and maximize the model performance was the accuracy score.
- Class imbalance in the training dataset was handled using the `class_weights` parameter in each model pipeline.
- The best model was the XGBoost Classifier pipeline.

***

## Trained Model Metrics

| Metric | Logistic Regression | Random Forest Classifier | XGBoost Classifier |
| :------------------ | :----------------------- | :------------------------ | :----------------- |
| Accuracy Score (1 and 0) | 0.8417 | 0.9313 | 0.9333 |
| Precision (1) | 0.8861 | 0.8970 | 0.8969 |
| Recall (1) | 0.8204 | 0.9899 | 0.9940 |
| F1 Score (1) | 0.8520 | 0.9412 | 0.9430 |
| Support Samples (1) | 84148 | 84148 | 84148 |

***

## Feature Importances: XGBoost Classifier

<img width="1000" height="500" alt="Feature importance plot" src="https://github.com/user-attachments/assets/cb77ca1b-afa5-4614-9e82-24aafd59919b" />

***

### Key Takeaways from Feature Importances

- The six most important features for predicting churn, in descending order, are:
  - Number of support calls
  - Monthly contract length
  - Female gender
  - Total spend
  - Payment delay
  - Age
- The remaining features did not contribute significantly to churn prediction.
- Surprisingly, the Male gender feature had zero feature importance in churn prediction.

***

# Streamlit Web App Development

- The best model (the XGBoost Classifier pipeline) was saved using Joblib.
- The feature importances dataframe was saved as a CSV file.
- Both resources were used to develop the Streamlit web app.
- The Streamlit Python script and all related resources can be found in the `Streamlit_App` folder.

**Click the link to launch the Streamlit app 👉👉👉**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-analysis-machine-learning-project-p64ajwiykxu4q.streamlit.app/)

***

# Business Recommendations

- Encourage quarterly or annual contracts instead of monthly contracts.
- Resolve customer enquiries and maintain support calls below five per month.
- Provide incentives to customers over 50 years old with total spending below $500.
- Target customers with tenure of less than 24 months and usage frequency of less than 10 times per month.

## About Me 👨‍🎓

I am a BSc Applied Mathematics and Statistics graduate with a passion for data science and analytics. This project reflects my ability to apply technical machine learning skills to real-world datasets.

