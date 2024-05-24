import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

sale_pred=pd.read_csv(r'advertising.csv')

sale_pred.head(5)

sale_pred.info()

sale_pred.describe()

plt.figure(figsize=(7, 5))
sns.regplot(x='Newspaper', y='Sales', data=sale_pred, line_kws={'color': 'red'})
plt.title('Newspaper Advertising Spend vs Sales')
plt.ylabel('Sales')
plt.xlabel('Newspaper Advertising Spend')
plt.show()

plt.figure(figsize=(7, 5))
sns.regplot(x='Radio', y='Sales', data=sale_pred, line_kws={'color': 'red'})
plt.title('Radio Advertising Spend vs Sales')
plt.ylabel('Sales')
plt.xlabel('Radio Advertising Spend')
plt.show()

plt.figure(figsize=(7, 5))
sns.regplot(x='Newspaper', y='Sales', data=sale_pred, line_kws={'color': 'red'})
plt.title('Newspaper Advertising Spend vs Sales')
plt.ylabel('Sales')
plt.xlabel('Newspaper Advertising Spend')
plt.show()

# Define features and target variable
X = sale_pred.drop(columns=['Sales'])  # Assuming 'sales' is the target variable
y = sale_pred['Sales']
X.head()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to DMatrix format for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters for XGBoost
params = {
    'max_depth': 3,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
}

# Train the XGBoost model
num_rounds = 100
xgb_model = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
y_pred = xgb_model.predict(dtest)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Feature importance
xgb.plot_importance(xgb_model)
plt.title('Feature Importances')
plt.show()
