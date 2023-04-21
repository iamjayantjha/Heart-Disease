import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
heart_data = pd.read_csv('heart.csv')
# print(heart_data.tail)
I = heart_data.drop(columns = 'target', axis=1)
O = heart_data['target']
I_train, I_test, O_train, O_test = train_test_split(I, O, test_size=0.3, stratify=O, random_state=2)
print(I.shape, I_train.shape, I_test.shape)
model = RandomForestClassifier(max_depth=None, random_state=0)
model.fit(I_train, O_train)
I_train_prediction = model.predict(I_train)
I_test_prediction = model.predict(I_test)
training_data_accuracy = accuracy_score(I_train_prediction, O_train)*100
test_data_accuracy = accuracy_score(I_test_prediction, O_test)*100
print('Accuracy on Training data for Random Forest : ', training_data_accuracy)
print('Accuracy on Test data for Random Forest : ', test_data_accuracy)
import pickle
#pickle.dump(model,open('model_random_forest.pkl','wb'))
model = LogisticRegression()
model.fit(I_train, O_train)
I_train_prediction = model.predict(I_train)
I_test_prediction = model.predict(I_test)
training_data_accuracy = accuracy_score(I_train_prediction, O_train)*100
test_data_accuracy = accuracy_score(I_test_prediction, O_test)*100
print('Accuracy on Training data for Logistic Regression : ', training_data_accuracy)
print('Accuracy on Test data for Logistic Regression : ', test_data_accuracy)
#pickle.dump(model,open('model_logistic_regression.pkl','wb'))
model = svm.SVC()
model.fit(I_train, O_train)
I_train_prediction = model.predict(I_train)
I_test_prediction = model.predict(I_test)
training_data_accuracy = accuracy_score(I_train_prediction, O_train)*100
test_data_accuracy = accuracy_score(I_test_prediction, O_test)*100
print('Accuracy on Training data for SVM : ', training_data_accuracy)
print('Accuracy on Test data for SVM : ', test_data_accuracy)
#pickle.dump(model,open('model_svm.pkl','wb'))
model = DecisionTreeClassifier()
model.fit(I_train, O_train)
I_train_prediction = model.predict(I_train)
I_test_prediction = model.predict(I_test)
training_data_accuracy = accuracy_score(I_train_prediction, O_train)*100
test_data_accuracy = accuracy_score(I_test_prediction, O_test)*100
print('Accuracy on Training data for Decision Tree : ', training_data_accuracy)
print('Accuracy on Test data for Decision Tree : ', test_data_accuracy)
#pickle.dump(model,open('model_decision_tree.pkl','wb'))