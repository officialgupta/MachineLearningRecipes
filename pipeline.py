from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn import tree # replaced with Kneighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# COLLECT TRAINING DATA

iris = datasets.load_iris()

X = iris.data #input FEATURES
y = iris.target #output LABELS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .5) #splits the data so some can be used for testing 50%

# TRAIN CLASSIFIER

# my_classifier = tree.DecisionTreeClassifier() #replaced with Kneighbor
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

# MAKE PREDICTIONS

predictions = my_classifier.predict(X_test)

# CHECK ACCURACY

print(accuracy_score(y_test, predictions))