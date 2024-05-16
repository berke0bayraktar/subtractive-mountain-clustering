import numpy as np
import math

class Centers:

    def __init__(self, X, R_A, R_B, E_UP, E_DOWN):

        # constants
        self.R_A = R_A
        self.R_B = R_B
        self.ALPHA = 4 / (R_A * R_A)
        self.BETA = 4 / (R_B * R_B)
        self.E_UP = E_UP
        self.E_DOWN = E_DOWN

        self.NUM_POINTS = X.shape[0]

        # algorithm data
        self.X = X
        self.P = np.empty(self.NUM_POINTS)
        self.CENTERS = []

    def __normalize_data(self):
        min_vals = self.X.min(axis=0)
        max_vals = self.X.max(axis=0)
        range_vals = max_vals - min_vals
        self.X = (self.X - min_vals) / range_vals

    def __compute_initial_potentials(self):
        squared_diff = np.sum((self.X[:, np.newaxis, :] - self.X[np.newaxis, :, :])**2, axis=2)
        exp_term = np.exp(-self.ALPHA * squared_diff)
        self.P = np.sum(exp_term, axis=1) - 1

    def __update_potentials(self, new_center_idx):
        x1 = self.X[new_center_idx, :]
        P1 = self.P[new_center_idx]
        squared_diffs = np.sum((self.X - x1)**2, axis=1)
        decay = P1 * np.exp(-self.BETA * squared_diffs)
        self.P = np.maximum(0, self.P - decay)

    def __is_good_center(self, i, PMAX):
        if self.P[i] > self.E_UP * PMAX: return True
        elif self.P[i] < self.E_DOWN * PMAX: return False
        else:
            squared_distances = np.sum((self.X[i, :] - self.X[self.CENTERS, :])**2, axis=1)
            min_dist = np.sqrt(np.min(squared_distances))
            return (min_dist / self.R_A) + (self.P[i] / PMAX) >= 1
        
    def __find_centers(self):

        # find first center
        first_center_idx = np.argmax(self.P)
        PMAX = self.P[first_center_idx]
        self.CENTERS.append(first_center_idx)
        self.__update_potentials(first_center_idx)

        # find rest of the centers
        mask = np.ones(self.NUM_POINTS, dtype=int)
        mask[self.CENTERS[0]] = 0 # remove the already decided initial center from candidates

        for _ in range(self.NUM_POINTS - 1):
            indices = np.nonzero(mask)[0]
            max_p_idx = indices[np.argmax(np.compress(mask, self.P))]

            if self.__is_good_center(max_p_idx, PMAX):
                self.CENTERS.append(max_p_idx)
                self.__update_potentials(max_p_idx)
                mask[max_p_idx] = 0

    def run(self):
        self.__normalize_data()
        self.__compute_initial_potentials()
        self.__find_centers()
        return self.CENTERS