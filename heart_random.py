import numpy as np    # For Importing Mathematical Tools
import matplotlib.pyplot as plt   # For Ploting Graphs
import pandas as pd    # For Importing Datasets
  
# Importing dataset
dataset = pd.read_csv('heart1.csv.xls')

# Setting X as Independent Columns
X= dataset.iloc[:,:-1].values
# All columns except the last one is selected 

# Setting Y as Dependent Columns
Y= dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split 
X_train,X_test,Y_train,Y_test=train_test_split (X,Y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# random forest 
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,
criterion = 'entropy', random_state = 0)
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

# naive bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


# SVM linear 
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
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

a=int(input("Enter age: "))
b=int(input("Enter gender: "))
c=int(input("Enter chest pain type (4 values): "))
d=int(input("Enter resting blood pressure: "))
e=int(input("Enter serum cholesterol(mg/dl): "))
f=int(input("Enter fasting blood sugar: "))
g=int(input("Enter resting electrocardiographic results (values 0,1,2): "))
i=int(input("Enter maximum heart rate achieved: "))
j=int(input("Enter exercise induced angina: "))
k=float(input("Enter oldpeak: "))
l=int(input("Enter the slope of the peak exercise ST segment:"))
m=int(input("Enter number of major vessels (0-3): "))
n=int(input("Enter thal(0 = normal; 1 = fixed defect; 2 = reversable defect):"))



Xnew = [[a,b,c,d,e,f,g,i,j,k,l,m,n]]
# make a prediction

ynew = classifier.predict(sc.transform(Xnew))
print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))

print("\n")

if ynew[0] == 1:
    print("YOU ARE SUFFERING FROM HEART DISEASE")
    print("\n")
    print("SOME REMEDIES ARE:")
    print("1. Stop smoking")
    print("2. Control your blood pressure")
    print("3. Check your cholesterol")
    print("4. Keep diabetes under control")
    print("5. Eat healthy foods")
    print("6. Maintain a healthy weight")
    print("7. Manage stress")
elif ynew[0] == 0:
    print("YOU ARE NOT SUFFERING FROM HEART DISEASE")


