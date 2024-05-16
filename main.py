from centers import Centers
from kmeans import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def main():
    
    # generate data
    data, _ = make_blobs(n_samples=20000, n_features=2, centers=50, cluster_std=0.2, random_state=0)
    
    # find centers
    mtn = Centers(data, R_A=0.1, R_B=0.15, E_UP=0.5, E_DOWN=0.15)
    centers = mtn.run()

    # cluster
    kmeans = KMeans(data, k=len(centers), iters=100, centers=data[centers])
    assignments = kmeans.run()
    
    # plot
    plt.scatter(data[:, 0], data[:, 1], c=assignments)
    plt.scatter(data[centers][:, 0], data[centers][:, 1], marker="X")
    plt.show()
    

if __name__ == "__main__":
    main()