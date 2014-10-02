import os
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score

os.chdir("/Users/BradLi/Documents/Data Science/Kaggle/Titanic")
writeFile = False

df_train = pd.read_csv("train.csv", header = 0)
df_test = pd.read_csv("test.csv", header = 0)
ids = df_test['PassengerId'].values
df = pd.merge(df_train, df_test, how = 'outer')

age_median = np.zeros(3)
fare_median = np.zeros(3)

for i in range(0,3):       
    age_median[i] = df[df.Pclass == i+1]['Age'].median()
    fare_median[i] = df[df.Pclass == i+1]['Fare'].median()
    df.loc[(df.Age.isnull()) & (df.Pclass == i+1), 'Age'] = age_median[i]
    df.loc[(df.Fare.isnull()) & (df.Pclass == i+1), 'Fare'] = fare_median[i]
    df.loc[df.Embarked.isnull(), 'Embarked'] = df['Pclass'].map({1:'C', 2:'S', 3:'Q'})
    

df['Gender'] = df['Sex'].map({'male':0, 'female':1})
df['Embark'] = df['Embarked'].map({'C':0, 'S':1, 'Q':0})
df['PclassD'] = df['Pclass'].map({3:3, 2:1, 1:1})
df['GenClass'] = df.Gender * df.Pclass
df['AgeClass'] = df.Age * df.Pclass
df['Fam'] = df.Parch**2 * df.SibSp**2
df = df.drop(['PassengerId','Cabin','Name','Ticket','Embarked','Sex','Pclass'], axis=1)
data = df.values

forest = RFC(n_estimators = 100)
cv_score = cross_val_score(forest, data[0:891,1::], data[0:891,0])
print "CV Score = ", cv_score.mean()
forest = forest.fit(data[0:891,1::], data[0:891,0])
output = forest.predict(data[891::,1::]).astype(int)

if writeFile == True:
    pfile = open("submission.csv","w+")
    p = csv.writer(pfile)
    p.writerow(['PassengerId','Survived'])
    p.writerows(zip(ids,output))
    
    pfile.close()


