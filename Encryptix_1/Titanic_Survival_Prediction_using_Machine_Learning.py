# Importing the Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection & Processing
# load the data from csv file to Pandas DataFrame
titanic_data = pd.read_csv(r'train.csv')

# printing the first 5 rows of the dataframe
titanic_data.head()

# number of rows and Columns
titanic_data.shape

# getting some informations about the data
titanic_data.info()

# check the number of missing values in each column
titanic_data.isnull().sum()

# Handling the Missing values
# drop the "Cabin" column from the dataframe
titanic_data = titanic_data.drop(columns='Cabin', axis=1)

# replacing the missing values in "Age" column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)

# finding the mode value of "Embarked" column
print(titanic_data['Embarked'].mode())
print(titanic_data['Embarked'].mode()[0])

# replacing the missing values in "Embarked" column with mode value
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)

# check the number of missing values in each column
titanic_data.isnull().sum()

# Data Analysis
# getting some statistical measures about the data
titanic_data.describe()

# finding the number of people survived and not survived
titanic_data['Survived'].value_counts()

# Data Visualization
sns.set()

# Making a pie chart for "Survived" column
survived_counts = titanic_data['Survived'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(survived_counts, labels=['Not Survived', 'Survived'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66FFB3'])
plt.title('Survived')
plt.show()
titanic_data['Sex'].value_counts()

# Making a pie chart for "Sex" column
sex_counts = titanic_data['Sex'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(sex_counts, labels=sex_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Sex')
plt.show()

# Number of survivors Gender wise
gender_survived_counts = titanic_data.groupby('Sex')['Survived'].value_counts().unstack()
plt.figure(figsize=(12, 6))
gender_survived_counts.plot(kind='pie', subplots=True, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'], legend=False)
plt.title('Survival by Gender')
plt.show()

# Making a pie chart for "Pclass" column
pclass_counts = titanic_data['Pclass'].value_counts()
plt.figure(figsize=(5, 5))
plt.pie(pclass_counts, labels=['Class 1', 'Class 2', 'Class 3'], autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Pclass')
plt.show()
sns.countplot(x='Pclass', hue='Survived', data=titanic_data, palette=['#66FFB3','#66b3ff'])

# Encoding the Categorical Columns
titanic_data['Sex'].value_counts()
titanic_data['Embarked'].value_counts()

# converting categorical Columns
titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
titanic_data.head()

# Separating features & Target
X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']
print(X)
print(Y)

# Splitting the data into training data & Test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Model Training
# Logistic Regression
model = LogisticRegression()

# training the Logistic Regression model with training data
model.fit(X_train, Y_train)

# Model Evaluation
# Accuracy Score

# accuracy on training data
X_train_prediction = model.predict(X_train)
print(X_train_prediction)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
print(X_test_prediction)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Confusion Matrix
cm = confusion_matrix(Y_test, model.predict(X_test))

# Precision, Recall, and F1-score
precision = precision_score(Y_test, model.predict(X_test))
recall = recall_score(Y_test, model.predict(X_test))
f1 = f1_score(Y_test, model.predict(X_test))

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# from sklearn.preprocessing import StandardScaler
# scaler=StandardScaler()
            # Pclass  Sex        Age    SibSp  Parch     Fare    Embarked  
input_data=(2,       1,        22,     1,      0,       7.25,    0)
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the data
# std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_reshaped)
print(prediction)


if (prediction[0] == 0):
  print("The Person does not Survived")

else:
  print("The Person has Survived")
