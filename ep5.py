from sklearn import datasets
from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

# COLLECT TRAINING DATA

iris = datasets.load_iris()

X = iris.data #input FEATURES
y = iris.target #output LABELS

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .5) #splits the data so some can be used for testing 50%

# TRAIN CLASSIFIER

my_classifier = ScrappyKNN()
my_classifier.fit(X_train, y_train)

# MAKE PREDICTIONS

predictions = my_classifier.predict(X_test)

# CHECK ACCURACY

print(accuracy_score(y_test, predictions))