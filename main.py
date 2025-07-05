import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import plot_tree, DecisionTreeClassifier

data = pd.read_csv("/Users/shyamsanogar/Documents/SkillCraft/TASK_03/bank+marketing/bank/bank.csv", sep=";", quotechar='"')

data['y'] = data['y'].map({'yes': 1, 'no': 0})

data_encoded = pd.get_dummies(data.drop('y', axis=1))

X = data_encoded
y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

plt.figure(figsize=(16, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()
