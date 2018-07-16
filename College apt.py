import sklearn
from sklearn import tree

#This data is from niche Carnegie Mellon Comp Sci Acceptance
#[gpa, act, sat]
X = [[2.5,22,1090],[3.6,24,1160],[4.0,36,1600],[2.2,34,1520],[3.3,34,1520],[3.1,29,1360],[3.4,25,1180],
	[4.0,22,1120],[3.5,20,1030],[3.4,29,1350],[3.7,30,1380]] 
	
Y =	['decline', 'accept', 'accept','decline', 'accept', 'accept', 'decline','decline', 'decline', 'accept','accept']
 

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)

#you can input your own data in the format of
#[gpa, act, sat]
prediction = clf.predict([[3.5,28,1250]])

print(prediction)
