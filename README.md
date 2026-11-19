**Click on the link to launch the Streamlit App 👉👉👉**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-analysis-machine-learning-project-p64ajwiykxu4q.streamlit.app/)

# Customer-Churn-Analysis-Machine-Learning-Project
The main focus of this project is to develop a machine learning model using Scikit learn to predict the customers who are likely to leave the business, and deploy the model using streamlit. The project includes Data Cleaning, EDA, model development using Logistic Regression, Random Forest and XGBOOST-Classifier, and developing streamlit App to deploy the best model amongst the three.

## Tech Stack

- **Python**
- **Pandas, NumPy** (data cleaning & processing)
- **Seaborn, Matplotlib, Plotly Express** (visualizations)
- **Scikit-learn** (modeling)
- **Streamlit, Joblib, Visual Studio Code** ( Web App development and deployment)
- **Jupyter Notebook**

# Data Cleaning
- The only data cleaning done on the dataset was to drop one row of missing values on row 199295 of the dataset.
  
# Exploratory Data Analysis (EDA)

The dataset was explored using the python visualization libraries such as: Seaborn, Matplotlib.pyplot, PlotlyExpress. 

**Key visualizations included:**

-  Count Plot showing **overall churn rate**
-  Histograms distributions of numerical features by churn.
-  Count plots distributions of categorical features by churn.
-  Box and Whisker plots of numerical features by churn.
-  Correlation heatmap for multi-colinearity of numerical features.
-  Correlation barbplot of numerical features to 'Churn'
***
**Key findings from the categorical features visuals:** 

- The dataset consisted of 56% samples of churners(1) as compared to the 44% Non-churners(0). ( This dataset imbalanced was handled during training of models).
- The Monthly Contract customers were found to be the most Churners in the Contract Length category: 90% Churners against 10% Non-Churners.
- The Female Gender consisted of 65% Churners against 35% Non-Churners.
***
**Key findings from the numerical features visuals:** 

- The Age feature revealed that: customers in the age range 50 to 65 appeared to churn at 84%.
- For Support Calls feature, the customers who made more than 4 support calls also churned at 84%.
- The Total Spend of less  than $500 is mostly associated with the churners, with churning rate of 90%.
- Payment Delay past 20 days is associated with 87% churners.
- Last Interaction days greater than 17 consists of 64% churning customers.
***
**key findings of the numerical features correlations to the churn***

- There was no multi-colinearity detected amongst the numerical features.
- The features most correlated to churn were:
- Support Calls (with correlation p = 0.5163),
- Total Spend,  p = -0.3697,
- Payment Delay, p = 0.3298.
- The rest of the numerical features had correlations less than 0.2.

# Machine learning Modelling.
The three machine learning models: Logistic Regression, Random Forest Classifier and XGBOOST Classifier, were fitted using the training data and the best model was selected based on the Recall of the Churn and the overall accuracy on the testing data.
***
## Preprocessing
- Data was plit into traning and testing data sets and strafified by Churn (the target labels).
- For **Logistic Regression preprocessor:** **OneHotIncoder( for Categorical features)** and **Standard scaler( for Numerical features)** were used to initiate the **ColumnTransformer**
- To prevent multi-colinearity, **OneHotIncoder(drop = 'first')** was used for Logistic Regression preprocessor. 
- Both **Random Forest and XGBOOST preprocessors** only required **OneHotIncoder(drop = None)** for preprocessing,
- since all predicting features are required for maximum learning in tree based models.
***
## Cross-validation and hyper-parameters tuning
- Pipelines coupled with the processors, and hyper-parameters for each model were initiated.
- **Coupling the preprocessors with the pipelines prevented data leakage during model training and cross-validation.**
- GridSearchCV fitted with training data to find the best model pipeline with the best hyper-parameter combinations.
- The metric to be maximized by the GridSearchCV was the **accuracy score.**
- Class imbalance in the training dataset was handled by managing the **class_weights** parameter in each of the trained model pipelines.
- **The best model was found to be the XGBOOST Classifier pipeline.**
***
## Trained models table of Metrics 
|Metric                |Logistic Regression       | Random Forest Classifier | XGBOOST Classifier |
| :------------------  | :----------------------- | :------------------------|:-------------------|
|Accuracy Score (1 & 0)| 0.8417                   | 0.9313                   |0.9333              |
|Precision  (1)        | 0.8861                   | 0.8970                   |0.8969              |
|Recall       (1)      | 0.8204                   | 0.9899                   |0.9940              |
|F1 Score       (1)    | 0.8520                   | 0.9412                   |0.9430              |
|Support Samples  (1)  |84148                     |84148                     |84148               |
***
## Feature Importances Plot of the best model (XGBOOST Classifier)
<img width="1000" height="500" alt="newplot(1)" src="https://github.com/user-attachments/assets/cb77ca1b-afa5-4614-9e82-24aafd59919b" />

***
### Key takeaways from feature importances

- The top 6 most important features in predicting Churn, in escending order are:
- **Number of Supporting Calls; Monthly(Contract length); Female(Gender); Total Spend; Payment Delay;  and Age.**
- The remainining feartures does not contribute significantly to Churn prediction.
- Suprisingly the Male(Gender) feature contribute zero importance in Churn prediction.
***
# Streamlit Web-App Development.
- The best model (XGBOOST Classifier Pipeline) saved using joblib.pkl;
- together with the feature importances dataframe saved as csv file;
- were used in the Streamlit Web-App development.
- **The streamlit Web-App  python script and all other resources used can be found under the Streamlit Folder.**

**Click on the link to launch the Streamlit App 👉👉👉**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-analysis-machine-learning-project-p64ajwiykxu4q.streamlit.app/)

