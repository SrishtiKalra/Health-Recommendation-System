
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn import model_selection
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import metrics



# Classifiers 
from sklearn.svm import NuSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingCVClassifier 

# Used to ignore warnings generated from StackingCVClassifier
import warnings
warnings.simplefilter('ignore')

df = pd.read_csv('heart1.csv.xls')
X= df.iloc[:,:-1].values
Y= df.iloc[:,-1].values



from sklearn.model_selection import KFold # import KFold


scores = []
cv = KFold(n_splits=10, random_state=42, shuffle=True)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]

    
#X_train,X_test,Y_train,Y_test=train_test_split (X,Y,test_size=0.2,random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Initializing Support Vector classifier
classifier1 = SVC(C = 50, degree = 1, gamma = "auto", kernel = "rbf", probability = True)

# Initializing Multi-layer perceptron  classifier
classifier2 = MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = 1000)

# Initialing Nu Support Vector classifier
classifier3 = NuSVC(degree = 1, kernel = "rbf", nu = 0.25, probability = True)

# Initializing Random Forest classifier
classifier4 = RandomForestClassifier(n_estimators = 500, criterion = "gini", max_depth = 10,
                                     max_features = "auto", min_samples_leaf = 0.005,
                                     min_samples_split = 0.005, n_jobs = -1, random_state = 1000)

# Initializing the StackingCV classifier
sclf = StackingCVClassifier(classifiers = [classifier1, classifier2, classifier3, classifier4],
                            shuffle = False,
                            use_probas = True,
                            cv = 5,
                            meta_classifier = SVC(probability = True))

# Create list to store classifiers
classifiers = {"SVC": classifier1,
               "MLP": classifier2,
               "NuSVC": classifier3,
               "RF": classifier4,
               "Stack": sclf}

# Train classifiers
for key in classifiers:
    # Get classifier
    classifier = classifiers[key]
    
    # Fit classifier
    classifier.fit(X_train, Y_train)
        
    # Save fitted classifier
    classifiers[key] = classifier
    
    # Get results
results = pd.DataFrame()
for key in classifiers:
    # Make prediction on test set
    y_pred = classifiers[key].predict_proba(X_test)[:,1]
    
    # Save results in pandas dataframe object
    results[f"{key}"] = y_pred

# Add the test set to the results object
results["Target"] = Y_test
'''
# Probability Distributions Figure
# Set graph style
sns.set(font_scale = 1)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})
    


# Plot
f, ax = plt.subplots(figsize=(13, 4), nrows=1, ncols = 5)

for key, counter in zip(classifiers, range(5)):
    # Get predictions
    y_pred = results[key]
    
    # Get AUC
    auc = metrics.roc_auc_score(Y_test, y_pred)
    textstr = f"AUC: {auc:.3f}"

    # Plot false distribution
    false_pred = results[results["Target"] == 0]
    sns.distplot(false_pred[key], hist=True, kde=False, 
                 bins=int(25), color = 'red',
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    # Plot true distribution
    true_pred = results[results["Target"] == 1]
    sns.distplot(results[key], hist=True, kde=False, 
                 bins=int(25), color = 'blue',
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    
    # These are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    # Place a text box in upper left in axes coords
    ax[counter].text(0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14,
                    verticalalignment = "top", bbox=props)
    
    # Set axis limits and labels
    ax[counter].set_title(f"{key} Distribution")
    ax[counter].set_xlim(0,1)
    ax[counter].set_xlabel("Probability")

# Tight layout
plt.tight_layout()

# Save Figure
plt.savefig("Probability Distribution for each Classifier.png", dpi = 1080)

y_pred=y_pred.round()

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)


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

ynew = classifiers["Stack"].predict(sc.transform(Xnew))
#print("X=%s, Predicted= %s" % (Xnew[0], ynew[0]))

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




