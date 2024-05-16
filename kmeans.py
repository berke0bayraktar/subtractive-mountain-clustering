import numpy as np

class KMeans:

    def __init__(self, data, k, iters=None, centers=None):

        self.X = data
        self.NUM_POINTS = self.X.shape[0]
        self.K = k
        self.iters = iters

        if centers is None:
            self.__init__centers()
        else:
            self.centers = centers

        self.assignments = np.zeros(self.NUM_POINTS)
        self.distances = np.zeros((self.NUM_POINTS, self.K))

    def __init_centers(self):
        raise NotImplementedError()
    
    def __calculate_distances(self):
        self.distances = np.sum((self.X[:, np.newaxis, :] - self.centers)**2, axis=2)

    def __assign_points_to_centers(self):
        self.assignments = np.argmin(self.distances, axis=1)
    
    def __update_centers(self):
        for i in range(self.K):
            self.centers[i, :] = np.mean(self.X[self.assignments == i, :], axis=0)

    def run(self):
        for _ in range(self.iters):
            self.__calculate_distances()
            self.__assign_points_to_centers()
            self.__update_centers()

        return self.assignments