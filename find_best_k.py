import numpy as np


def find_best_k(X_train, y_train, X_test, y_test):
    n1 = len(X_train)
    best_k = 1
    max_acc = 0
    for k in range(1, n1 + 1):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = (y_pred == y_test).mean()
        if acc > max_acc or (acc == max_acc and k < best_k):
            max_acc = acc
            best_k = k
    return best_k
