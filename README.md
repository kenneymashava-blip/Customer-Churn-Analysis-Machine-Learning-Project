# Customer-Churn-Analysis-Machine-Learning-Project
The main focus of this project is to develop a machine learning model using Scikit learn to predict the customers who are likely to leave the business, and deploy the model using streamlit. The project includes Data Cleaning, EDA, model development using Logistic Regression, Random Forest and XGBOOST-Classifier, and developing streamlit App to deploy the best model amongst the three.


## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy** (data cleaning & processing)
- **Seaborn, Matplotlib, Plotly Express** (visualizations)
- **Scikit-learn** (modeling)
- **Streamlit, Visual Studio Code** ( Web App development and deployment)
- **Jupyter Notebook**


# Results
### Table of Metrics

|Metric                |Logistic Regression       | Random Forest Classifier | XGBOOST Classifier |
| :------------------  | :----------------------- | :------------------------|:-------------------|
|Accuracy Score (1 & 0)| 0.8417                   | 0.9313                   |0.9333              |
|Precision  (1)        | 0.8861                   | 0.8970                   |0.8969              |
|Recall       (1)      | 0.8204                   | 0.9899                   |0.9940              |
|F1 Score       (1)    | 0.8520                   | 0.9412                   |0.9430              |
|Support Samples  (1)  |84148                     |84148                     |84148               |

# Feature Importances Plot of the best model (XGBOOST Classifier)

<img width="1000" height="500" alt="newplot(1)" src="https://github.com/user-attachments/assets/cb77ca1b-afa5-4614-9e82-24aafd59919b" />

