import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import classification_utils as classification_utils


def train(X_train, Y_train, X_test, Y_test):
    svm = SVC(probability=True)
    svm.fit(X_train, Y_train)
    acc = np.sum(svm.predict(X_test) == Y_test) / len(Y_test)
    print(f"Accuracy test: {acc}")
    f1 = f1_score(Y_test, svm.predict(X_test), average=classification_utils.getF1ScoreAveraging(Y_test))
    print(f"F1-Score test: {f1}")
    return svm
