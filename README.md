# Titanic-Survival-Predictor
# Predicting Titanic passenger survival using four ML models with feature-based analysis and visualizations.
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# load titanic dataset
df = sns.load_dataset('titanic')

# select and clean relevant columns (preprocess the data,keep only important features)
df = df[['survived', 'pclass', 'sex', 'age', 'fare', 'embarked']]
# drop rows with missing values 
df.dropna(inplace=True)

# Encode categorical variables (convert categorical data to numerical)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q':2 })

# FEATURES AND TARGET
X = df[['pclass', 'sex', 'age', 'fare', 'embarked']]
y = df['survived']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1 Train decision tree 
tree_model = DecisionTreeClassifier(max_depth=4, random_state=0)
tree_model.fit(X_train, y_train)
# make precictions 
y_pred_tree = tree_model.predict(X_test)
# evaluate model
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_tree))

# visualize Decision Tree 
plt.figure(figsize=(16,10))
plot_tree(tree_model, feature_names=X.columns, class_names=["Did Not Survive", "Survived"], filled=True)
plt.title("Titaniv Survival Decision Tree")
plt.show()

# 2 Random Forest 
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf =rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))


# 3 Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# 4 Logistic Regression
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X_train, y_train)
y_pred_logreg = logreg_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_logreg))


# visualization
sns.barplot(x='sex', y='survived', data=df)
plt.title("Survival Rate by Gender")
plt.show()

sns.histplot(df[df['survived'] == 1]['age'], bins=20, kde=True)
plt.title("Age Distribution of survivors")
plt.xlabel("Age")
plt.show()

