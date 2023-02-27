import numpy as np
import pandas as pd


class Knn:
    # k-Nearest Neighbor class object for classification training and testing
    def __init__(self):
        self.x = []
        self.y = []

    def fit(self, x, y):
        # Save the training data to properties of this class
        self.x = np.array(x)
        self.y = np.array(y)

    def e_distance(pts, ref_pts):
        X_new = np.sqrt(np.sum(np.square(pts - ref_pts)))
        return X_new

    def knn_prediction(distance, y, K=1):
        return pd.DataFrame(y[distance.loc[distance.distance.rank() <= K].index]).mode()

    def predict(self, x, k):
        y_hat = []  # Variable to store the estimated class label for
        # Calculate the distance from each vector in x to the training data
        dis = pd.DataFrame(self.x)
        x = np.array(x)
        for x_ind in range(len(x)):
            a = np.zeros(np.shape(self.x)) + x[x_ind]
            dis.loc[:, "distance"] = np.sqrt(np.sum((self.x - a) ** 2, axis=1))
            y_hat.append(
                pd.DataFrame(self.y[dis.loc[dis.distance.rank() <= k].index]).mode()
            )
        return np.array([y_hat[i].iloc[0, 0] for i in range(len(y_hat))]).astype(int)


# Metric of overall classification accuracy
#  (a more general function, sklearn.metrics.accuracy_score, is also available)
def accuracy(y, y_hat):
    nvalues = len(y)
    accuracy = sum(y == y_hat) / nvalues
    return accuracy
