1. Similarity matrix construction

python similarityMeasure.py mode alpha convergence

	Require the graph data file(default:"graph.txt") is in the same directory.
	Output: a similarity matrix file(default:"similarity.txt"), a graph information file(default:"graphInfo.npy")
	
	mode: 1 -> jaccard similarity, 2 -> ppr similarity
	alpha: ppr alpha parameter(only supported for mode 2)
	convergence: ppr convergence convergence threshold(only supported for mode 2)

	The default mode -> 2
	The default alpha -> -.1
	The default convergence threshold -> 1e-05

2. Kmeans clustering

python kmeans.py k

	Require the input data file(default:"similarity.txt") is in the same directory.
	Output: a clustering result file(default:"clustering.npy")

	k: The specified cluster number
	
	The default k -> 5

3. Clustering benchmark

python clusteringBenchmark.py

	Require the graph information file(default:"graphInfo.npy"), the labels file(default:"labels.txt") and the clustering result file(default:"clustering.npy") are in the same directory
	No other parameters needed.
	Output: the purity, the entropy of the clustering result, the entropy of the labels, the NMI are printed in the terminal
