from sklearn import tree

# COLLECT TRAINING DATA

# 1 = smooth : 0 = bumpy : REAL VALUES NEEDED
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 = Apple : 1 = Orange : REAL VALUES NEEDED
labels = [0, 0, 1, 1]

# TRAIN CLASSIFIER

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

# MAKE PREDICTIONS

print(clf.predict([[200, 1]]))