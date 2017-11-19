import numpy as np
import sys, time

GRAPH_FILENAME = "Graph.txt"
GRAPHINFO_OUTPUT_FILENAME = "graphInfo.npy"
SIMILARITY_OUTPUT_FILANAME = "similarity.txt"

DEFAULT_MODE = 2
DEFAULT_ALPHA = 0.1
DEFAULT_CONVERGENCE = 1e-05

def append_neighbor(graph, node, neighborNode):
    if node in graph:
        graph[node].add(neighborNode)
    else:
        graph[node] = {neighborNode}


def load_graph():
    graph = dict()
    f = open(GRAPH_FILENAME, 'r')
    for line in f:
        [node1Str, node2Str] = line.strip().split(' ')
        node1 = int(node1Str)
        node2 = int(node2Str)
        append_neighbor(graph, node1, node2)
        append_neighbor(graph, node2, node1)
    f.close()
    return graph

# Index for every node
def get_node_index(graph):
    nodesIndexes = dict()
    for (index, node) in list(enumerate(graph.keys())):
        nodesIndexes[node] = index
    return nodesIndexes

# Get the transition probability matrix
def get_transition_matrix(graph, nodesIndexes):
    nodesNum = len(graph)
    transitionMatrix = np.zeros((nodesNum, nodesNum))
    for node, neighborNodes in graph.items():
        neighborNum = len(neighborNodes)
        transitionProb = 1/neighborNum
        for neighborNode in neighborNodes:
            transitionMatrix[nodesIndexes[neighborNode]][nodesIndexes[node]] = transitionProb
    return transitionMatrix

# Personalize Page-Rank from the given node
def ppr(graph, nodesIndexes, transitionMatrix, node, alpha, convergence):
    initialVector = transitionMatrix[:, nodesIndexes[node]]
    proximityVector = initialVector.copy()

    while True:
        nextProximityVector = (1 - alpha)*transitionMatrix.dot(proximityVector) + alpha*initialVector
        if np.linalg.norm(nextProximityVector - proximityVector) < convergence:
            break
        else:
            proximityVector = nextProximityVector

    return proximityVector


def jaccard(graph, nodesIndexes, transitionMatrix, node):
    nodes = list(graph.keys())
    nodesNum = len(nodes)
    neighborNodes = graph[node]
    similarityVector = np.zeros(nodesNum)

    for i in range(nodesNum):
        tmpNeighborNodes = graph[nodes[i]]
        similarityVector[i] = len(tmpNeighborNodes.intersection(neighborNodes))/len(tmpNeighborNodes.union(neighborNodes))

    return similarityVector


if __name__ == '__main__':
    # argv:[pyFileName, mode, alpha, convergence]
    # mode: 1 -> jaccard similarity; 2 (default) -> ppr similarity
    # alpha (only for mode 2): 0.1 (default)
    # convergence threshold (only for mode 2): 1e-05 (default)
    mode = DEFAULT_MODE
    alpha = DEFAULT_ALPHA
    convergence = DEFAULT_CONVERGENCE
    lenSysArgv = len(sys.argv)
    if lenSysArgv > 4:
        print("Invalid argument number!")
        exit()

    if lenSysArgv > 1:
        try:
            mode = int(sys.argv[1])
        except Exception as e:
            print("Invalid argument(mode) exception: %s" % e)
            exit()

        if mode == 1:
            if lenSysArgv > 2:
                print("Warning: Jaccard mode doesn't need alpha and convergence parameters.")
        elif mode == 2:
            if lenSysArgv > 2:
                try:
                    alpha = float(sys.argv[2])
                    if alpha < 0 or alpha > 1:
                        print("The alpha parameter should be in [0, 1]")
                        exit()
                except Exception as e:
                    print("Invalid argument(alpha) exception: %s" % e)
                    exit()

            if lenSysArgv > 3:
                try:
                    convergence = float(sys.argv[3])
                except Exception as e:
                    print("Invalid argument(convergence) exception: %s" % e)
                    exit()
        else:
            print("Unsupported mode: %d" % mode)
            exit()

    startTime1 = time.time()
    graph = load_graph()
    endTime1 = time.time()
    print("Time cost for loading the graph is %.6f seconds" % (endTime1 - startTime1))

    startTime2 = time.time()
    nodesNum = len(graph)
    nodesIndexes = get_node_index(graph)
    transitionMatrix = get_transition_matrix(graph, nodesIndexes)
    similarityMatrix = np.zeros((nodesNum, nodesNum))
    for node in graph:
        if mode == 1:
            similarityMatrix[nodesIndexes[node]] = jaccard(graph, nodesIndexes, transitionMatrix, node)
        else:
            similarityMatrix[nodesIndexes[node]] = ppr(graph, nodesIndexes, transitionMatrix, node, alpha, convergence)
    endTime2 = time.time()
    print("Time cost for calculating the similarity matrix is %.4f seconds" % (endTime2 - startTime2))

    # Save the graph info
    graphInfo = dict()
    graphInfo['graph'] = graph
    graphInfo['nodesIndexes'] = nodesIndexes
    np.save(GRAPHINFO_OUTPUT_FILENAME, graphInfo)

    # Save the similarity matrix
    np.savetxt(SIMILARITY_OUTPUT_FILANAME, similarityMatrix)
