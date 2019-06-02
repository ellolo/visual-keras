import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from visual_keras.utils import get_layers_output


def compute_knn(query_vecs, index_vecs, k=5):
    # computing pairwise similarity
    sim_matrix = euclidean_distances(query_vecs, index_vecs)

    # getting top-k
    queries_topk = np.zeros((query_vecs.shape[0], index_vecs.shape[0]), dtype=int)
    for i in range(query_vecs.shape[0]):
        sims = sim_matrix[i]
        queries_topk[i] = sims.argsort()
    return queries_topk[:, :k]