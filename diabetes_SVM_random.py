import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('diabetes.csv')
X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,-1].values

# splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split (X,Y,test_size=1/3,random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# SVM linear 
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

'''

# KNN

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 2)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

# random forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,
criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


#SVM rbf
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# decision tree
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "gini", random_state = 0)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# logistic
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
'''


a=int(input("Enter number of pregnancies: "))
b=int(input("Enter plasma glucose concentration: "))
c=int(input("Enter diastolic blood pressure(mm Hg): "))
d=int(input("Enter riceps skin fold thickness(mm): "))
e=int(input("Enter insulin(mu U/ml): "))
f=float(input("Enter body mass index: "))
g=float(input("Enter diabetes pedigree function: "))
h=int(input("Enter age: "))


Xnew = [[a,b,c,d,e,f,g,h]]
# make a prediction


ynew = classifier.predict(sc.transform(Xnew))
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

print("\n")

if ynew[0] == 1:
    print("YOU ARE SUFFERING FROM DIABETES")
    print("\n")
    print("SOME REMEDIES ARE:")
    print("1. Exercise Regularly")
    print("2. Control Your Carb Intakey")
    print("3. Increase Your Fiber Intake")
    print("4. Drink Water and Stay Hydrated")
    print("5. Implement Portion Control")
    print("6. Control Stress Levels")
    print("7. Monitor Your Blood Sugar Levels")
elif ynew[0] == 0:
    print("YOU ARE NOT SUFFERING FROM DIABETES")

