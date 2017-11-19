import numpy as np
import sys, math, time

LABELS_FILENAME = "Labels.txt"

def load_labels():
    labels = dict()
    nodesLabels = dict()
    f = open(LABELS_FILENAME, 'r')
    for line in f:
        [nodeStr, labelStr] = line.strip().split(' ')
        node = int(nodeStr)
        label = int(labelStr)
        nodesLabels[node] = label
        if label not in labels:
            labels[label] = set({node})
        else:
            labels[label].add(node)
    f.close()
    return labels, nodesLabels


def get_filtered_clusters(clusters, nodes, nodesLabels):
    newClusters = dict()
    for centroid in clusters:
        labeledNodes = []
        for nodeIndex in clusters[centroid]:
            node = nodes[nodeIndex]
            if node in nodesLabels:
                labeledNodes.append(node)
        if labeledNodes != []:
            newClusters[centroid] = labeledNodes
    return newClusters


def get_purity(clusters, nodesLabels, nodesNum):
    correctCounts = 0
    for centroid in clusters:
        tmpNodesLabels = [nodesLabels[node] for node in clusters[centroid]]
        (labels, counts) = np.unique(tmpNodesLabels, return_counts=True)
        correctCounts += max(counts)
    purity = correctCounts/nodesNum
    return purity


def get_entropy(clusters, nodesNum):
    entropy = 0
    for centroid in clusters:
        prob = len(clusters[centroid])/nodesNum
        if prob != 0:
            entropy -= prob*math.log2(prob)
    return entropy


# Normalized mutual information
def get_NMI(clusters, labels, clustersEntropy, labelsEntropy, nodesNum):
    mutualInfo = 0
    for centroid in clusters:
        tmpClusteredNodes = clusters[centroid]
        tmpClusteredNodesNum = len(tmpClusteredNodes)
        for label in labels:
            tmpLabeledNodes = labels[label]
            tmpLabeledNodesNum = len(tmpLabeledNodes)
            intersectionNodesNum = len(tmpLabeledNodes.intersection(tmpClusteredNodes))

            if intersectionNodesNum != 0:
                mutualInfo += (intersectionNodesNum/nodesNum) * math.log2((nodesNum*intersectionNodesNum)/(tmpLabeledNodesNum*tmpClusteredNodesNum))

    return mutualInfo*2/(clustersEntropy + labelsEntropy)


if __name__ == '__main__':
    startTime1 = time.time()
    # Load relevant data files
    clusters = np.load("clusters.npy").item()
    graphInfo = np.load("graphInfo.npy").item()
    nodes = list(graphInfo['graph'].keys())
    nodesIndexes = graphInfo['nodesIndexes']
    labels, nodesLabels = load_labels()
    endTime1 = time.time()
    print("Time cost for loading the data file is %.4f seconds" % (endTime1 - startTime1))


    startTime2 = time.time()
    # Compute the evaluation criterions
    nodesNum = len(nodesLabels)
    newClusters = get_filtered_clusters(clusters, nodes, nodesLabels)
    purity = get_purity(newClusters, nodesLabels, nodesNum)
    clustersEntropy = get_entropy(newClusters, nodesNum)
    labelsEntropy = get_entropy(labels, nodesNum)
    nmi = get_NMI(newClusters, labels, clustersEntropy, labelsEntropy, nodesNum)
    endTime2 = time.time()

    print("="*58)
    print("| purity | clustered entropy | labeled entropy |   NMI   |")
    print("-"*58)
    print("| %6.4f | %17.4f | %15.4f | %6.4f |" % (purity, clustersEntropy, labelsEntropy, nmi))
    print("="*58)
    print("Time cost for computing the evaluation criterions is %.6f seconds" % (endTime2 - startTime2))
