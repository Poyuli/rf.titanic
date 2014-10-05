import os
import csv
import numpy as np
import pandas as pd
from sklearn import svm, grid_search
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score

os.chdir("/Users/BradLi/Documents/Data Science/Kaggle/Titanic")
writeFile = True
model = "SVM"

# Merging the training and test sets
df_train = pd.read_csv("train.csv", header = 0)
df_test = pd.read_csv("test.csv", header = 0)
ids = df_test['PassengerId'].values
df = df_train.append(df_test, ignore_index = True)
cols = df.columns.tolist()
cols = cols[-2:-1] + cols[:-2] + cols[-1:]
df = df[cols]

age_median = np.zeros(3)
fare_median = np.zeros(3)

# Discovering the correlation between missing features and others
# Missing features are Age, Fare, Embarked
embark = df.groupby('Embarked')
print "Pclass stats grouped by Embarked"
print embark.describe().Pclass,"\n"
print "Correlation between Age and other features"
print df.corr().Age,"\n"
print "Correlation between Fare and other features"
print df.corr().Fare,"\n"

# Filling the missing features
# Age and Fare are negatively correlated to Pclass
for i in range(0,3):       
    age_median[i] = df[df.Pclass == i+1]['Age'].median()
    fare_median[i] = df[df.Pclass == i+1]['Fare'].median()
    df.loc[(df.Age.isnull()) & (df.Pclass == i+1), 'Age'] = age_median[i]
    df.loc[(df.Fare.isnull()) & (df.Pclass == i+1), 'Fare'] = fare_median[i]
    df.loc[df.Embarked.isnull(), 'Embarked'] = df['Pclass'].map({1:'C', 2:'S', 3:'Q'})
    
# Exploring Embarked to make reasonable feature engineering
print "Fare stats wrt different Embarked:"
print embark.describe().Fare

# Feature engineering: mapping and discretization
df['Gender1'] = df['Sex'].map({'male':1, 'female':0})
df['Gender2'] = df['Sex'].map({'male':0, 'female':1})
df['Embark1'] = df['Embarked'].map({'C':1, 'S':0, 'Q':0})
df['Embark2'] = df['Embarked'].map({'C':0, 'S':1, 'Q':1})
df['AgeD'] = pd.cut(df.Age, [0,20,40,80], labels = ['A','B','C'])
df['Age1'] = df['AgeD'].map({'A':1, 'B':0, 'C':1})
df['Age2'] = df['AgeD'].map({'A':0, 'B':1, 'C':0})
df['AgeGender'] = df.Age1 * df.Gender1
df['ClassGender'] = df.Pclass * df.Gender1
df['Alone1'] = pd.cut(df.Parch, [0,0.5,9], labels = [1,0], include_lowest = True)
df['Alone2'] = pd.cut(df.SibSp, [0,0.5,8], labels = [1,0], include_lowest = True)
df['Alone'] = df.Alone1 * df.Alone2

df = df.drop(['PassengerId','Cabin','Name','Ticket','Embarked','Sex','Age','AgeD','Alone1','Alone2','Parch','SibSp'], axis=1)
if model == "SVM":
    df = (df - df.min()) / (df.max() - df.min())
data = df.values

# Random forest or SVM training
if model == "RF":
    forest = RFC(n_estimators = 100)
    cv_score = cross_val_score(forest, data[0:891,1::], data[0:891,0], cv=10)
    print "CV Score = ", cv_score.mean(),"\n"
    forest = forest.fit(data[0:891,1::], data[0:891,0])
    output = forest.predict(data[891::,1::]).astype(int)
    print "Feature importances:"
    features = df.columns.tolist()[1:]
    print zip(features, forest.feature_importances_)
    
elif model == "SVM":
    svc = svm.SVC()
    param = {'C':[1e4,1e3,1e2,1e1,1e0,1e-1,1e-2,1e-3], 'gamma':[1e-1,1e0,1e1,1e2,1e3,1e4,1e5], 'kernel':['rbf']}
    svc = grid_search.GridSearchCV(svc, param, cv=10)
    svc.fit(data[0:891,1::], data[0:891,0])
    print "Optimized parameters:"
    print svc.best_estimator_
    output = svc.predict(data[891::,1::]).astype(int)
    print "Best CV Score = ",svc.best_score_

# Writing predicted results to csv file
if writeFile == True:
    pfile = open("submission.csv","w+")
    p = csv.writer(pfile)
    p.writerow(['PassengerId','Survived'])
    p.writerows(zip(ids,output))  
    pfile.close()


