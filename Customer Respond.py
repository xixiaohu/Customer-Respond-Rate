import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt
import numpy as np

#read data from csv
df = pd.read_csv("finaldata.csv")
#convert categorical data into integers
df['RFA_2A'] = df['RFA_2A'].astype('category')
df["RFA_2A"] = df["RFA_2A"].cat.codes

#construct x and y
data = df.loc[:,['AVGGIFT', 'LASTGIFT', 'RFA_2F', 'RFA_2A', 'LASTDATE', 'FISTDATE',  'INCOME',  'WEALTH_INDEX']]
x = data.iloc[:, [0, 1, 2, 3, 5, 6, 7]].values
y = df['TARGET_B'].values

#standardlize data
scaler = StandardScaler()
x = scaler.fit_transform(x)
#split datasets into train and test dataset
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=23)

#Linear SVC
clf = LinearSVC()
clf.fit(X_train, Y_train)
y_clf=clf.predict(X_test)

score_clf = y_clf
pos = pd.get_dummies(Y_test).as_matrix()
pos = pos[:,  1]
npos = np.sum(pos)
index_clf = np.argsort(score_clf)
index_clf = index_clf[:: -1]
sort_pos_clf = pos[index_clf]
cpos_clf = np.cumsum(sort_pos_clf)
rappel_clf = cpos_clf / npos


#KNeighbors(k = 3)
neigh = KNeighborsClassifier(n_neighbors=3) 
neigh.fit(X_train, Y_train)
y_neigh=neigh.predict_proba(X_test)

score_neigh = y_neigh[:, 1]
index_neigh = np.argsort(score_neigh)
index_neigh = index_neigh[:: -1]
sort_pos_neigh = pos[index_neigh]
cpos_neigh = np.cumsum(sort_pos_neigh)
rappel_neigh = cpos_neigh / npos

#Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
y_gnb=gnb.predict_proba(X_test)

score_gnb = y_gnb[:, 1]
index_gnb = np.argsort(score_gnb)
index_gnb = index_gnb[:: -1]
sort_pos_gnb = pos[index_gnb]
cpos_gnb = np.cumsum(sort_pos_gnb)
rappel_gnb = cpos_gnb / npos

n = Y_test.shape[0]
taille = np.arange (start = 1,  stop = n+1,  step = 1)
taille = taille / n

#plot the performance
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.plot(taille,  taille,  color='peachpuff', ls = '--',  label='Random')
plt.plot(taille,  rappel_clf,  color='mediumturquoise', label='Linear SVC')
plt.plot(taille,  rappel_neigh,  color='orangered', label='KNeighbors')
plt.plot(taille,  rappel_gnb,  color='lightskyblue', label='Naive Bayes')
plt.xlabel('% Customers')
plt.ylabel('Positive Responses')
plt.legend()
plt.title('Cumulative Gains Chart of 3 Classifier')
plt.show()



