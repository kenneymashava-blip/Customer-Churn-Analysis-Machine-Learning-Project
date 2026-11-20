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
- Monthly contract customers contribute 22% to the business; with the percentage churn of 20% and only 2% non churn.
- Both Annual and Quartely contract customers contribute 39% each to the business, with 18% percentage churn and 21% non churn.
- The Female gender customers contribute approximatley 45% to the business, with 29% percentage churn and 16% non churn.
- The Male gender customers contribute approximatley 56% to the business, with 27% percentage churn and 29% non churn.
- The percentage churn by Subscription Type appears to be approximatley equivalent across all the subscription types.
- Basic and Standard subcriptions churn at 19% while Premium churn at 18%.
- Standard and Premium subsciptions have 15% non churn customers while Basic have 14%.
***
**Key findings from the numerical features visuals:** 
- Customers over the Age of 50 contribute 23% to the business, with 19% ending up Churning and 4% not churning.
- Customers who make 4 or more support calls a month contribute 45% to the business, with 38% ending up churning while 7% remain with the business.
- Customers with a Total Spend less than $500 contribute 29% to the business, with 26% ending up churning and only 3% remaining with the business.
- Customers who delay payments by more 20 days or more, contribute 26% to the business, with 23% churners and only 3% non churners.
- Customers who do not interact with the business for 17 or more days in a month contribute to 38% to the business,
- with 24% ending up churning while 14% remain with the business.
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

- The top 6 most important features in predicting Churn, in descending order are:
- **Number of Supporting Calls; Monthly(Contract length); Female(Gender); Total Spend; Payment Delay;  and Age.**
- The remaining feartures does not contribute significantly to Churn prediction.
- Suprisingly the Male(Gender) contribute zero feature importance in Churn prediction.
***
# Streamlit Web-App Development.
- The best model (XGBOOST Classifier Pipeline) saved using joblib.pkl;
- together with the feature importances dataframe saved as csv file;
- were used in the Streamlit Web-App development.
- **The streamlit Web-App  python script and all other resources used can be found under the Streamlit_App Folder.**

**Click on the link to launch the Streamlit App 👉👉👉**
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-analysis-machine-learning-project-p64ajwiykxu4q.streamlit.app/)
***
# Business recomendations
- Encourage Quarterly or Annual contracts instead of Monthly.
- Resolve customers enquiries and maintaning support calls below 5. 
- Provide incentives to customers over the age of 50 and total spend of less than $500.
- Target customers with less than 24 months ternure and less than 10 usage frequency per month.

👨‍🎓 About Me¶
I'm a Bsc Applied Mathematics and Statistics graduate with passion for data science and analytics. This project reflects my ability to combine technical machine learning skills with real-world business thinkin. Looking for roles in Data Analysis or Data Science.
