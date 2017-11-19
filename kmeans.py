import numpy as np
import sys, time

DATA_INPUT_FILENAME = "similarity.txt"
CLUSTERS_OUTPUT_FILENAME = "clusters.npy"

DEFAULT_K = 5

def kmeans(data, k):
    # initial random points
    centroids = [data[i] for i in np.random.choice(data.shape[0], k, replace=False)]
    clusters = dict()

    while True:
        clusters = dict()
        for centroid in centroids:
            clusters[tuple(centroid)] = set()

        # Calculate the nearest centroid for every point
        for index, point in enumerate(data):
            minDistance = float('inf')
            nearestCentroid = centroids[0]
            for centroid in centroids:
                distance = np.linalg.norm(point - centroid)
                if distance < minDistance:
                    minDistance = distance
                    nearestCentroid = centroid
            clusters[tuple(nearestCentroid)].add(index)

        # Update centroids
        newCentroids = []
        for centroid in centroids:
            neighborPointsIndexes = clusters[tuple(centroid)]
            sumDistance = 0
            for index in neighborPointsIndexes:
                sumDistance += data[index]
            newCentroids.append(sumDistance/len(neighborPointsIndexes))

        # Stop condition
        if np.array_equal(centroids, newCentroids):
            break
        else:
            centroids = newCentroids
    return clusters


if __name__ == '__main__':
    # argv:[pyFileName, k]
    # k: 5 (default)
    k = DEFAULT_K
    lenSysArgv = len(sys.argv)

    if lenSysArgv > 2:
        print("Invalid argument number!")
        exit()

    if lenSysArgv > 1:
        try:
            k = int(sys.argv[1])
        except Exception as e:
            print("Invalid argument(k) exception: %s" % e)
            exit()

    data = np.loadtxt(DATA_INPUT_FILENAME)
    startTime = time.time()
    clusters = kmeans(data, k)
    endTime = time.time()
    print("Time cost for K-Means clustering is %.4f seconds" % (endTime - startTime))
    np.save(CLUSTERS_OUTPUT_FILENAME, clusters)
