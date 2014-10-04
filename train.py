import os
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score

os.chdir("/Users/BradLi/Documents/Data Science/Kaggle/Titanic")
writeFile = True

df_train = pd.read_csv("train.csv", header = 0)
df_test = pd.read_csv("test.csv", header = 0)
ids = df_test['PassengerId'].values
df = pd.merge(df_train, df_test, how = 'outer')

age_median = np.zeros(3)
fare_median = np.zeros(3)

embark = df.groupby('Embarked')
print embark.describe().Pclass

for i in range(0,3):       
    age_median[i] = df[df.Pclass == i+1]['Age'].median()
    fare_median[i] = df[df.Pclass == i+1]['Fare'].median()
    df.loc[(df.Age.isnull()) & (df.Pclass == i+1), 'Age'] = age_median[i]
    df.loc[(df.Fare.isnull()) & (df.Pclass == i+1), 'Fare'] = fare_median[i]
    df.loc[df.Embarked.isnull(), 'Embarked'] = df['Pclass'].map({1:'C', 2:'S', 3:'Q'})

df['Gender1'] = df['Sex'].map({'male':1, 'female':0})
df['Gender2'] = df['Sex'].map({'male':0, 'female':1})
df['Embark1'] = df['Embarked'].map({'C':1, 'S':0, 'Q':0})
df['Embark2'] = df['Embarked'].map({'C':0, 'S':1, 'Q':0})
df['Embark3'] = df['Embarked'].map({'C':0, 'S':0, 'Q':1})
df['AgeGender'] = df.Age * df.Gender1
df['ClassGender'] = df.Pclass * df.Gender1
df['Fam'] = df.Parch**2 * df.SibSp**2

df = df.drop(['PassengerId','Cabin','Name','Ticket','Embarked','Sex'], axis=1)
data = df.values

forest = RFC(n_estimators = 100)
cv_score = cross_val_score(forest, data[0:891,1::], data[0:891,0], cv=10)
print "CV Score = ", cv_score.mean()
forest = forest.fit(data[0:891,1::], data[0:891,0])
output = forest.predict(data[891::,1::]).astype(int)

if writeFile == True:
    pfile = open("submission.csv","w+")
    p = csv.writer(pfile)
    p.writerow(['PassengerId','Survived'])
    p.writerows(zip(ids,output))
    
    pfile.close()


