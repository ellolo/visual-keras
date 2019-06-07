import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def compute_knn(query_vecs, index_vecs, k=5):
    """
    Finds k-nearest neighbors for each of the input quer_vecs vectors, out of the input index_vecs vectors.
    Euclidean distance is used as similarity measure to compute top k.

    Parameters
    ----------
    query_vecs : numpy.array
        query vectors for which to compute top k. Shape must be (num_vectors, vec_length).
    index_vecs : numpy.array
        query vectors from which to extract top k. Shape must be (num_vectors, vec_length).
    k : int
        number of top similar index_vecs to return (default is 5)

    Returns
    -------
    numpy.array
        Array of the indexes of the top k in the index_vecs. Array is sorted by descending similarity
    """

    # computing pairwise similarity
    sim_matrix = euclidean_distances(query_vecs, index_vecs)

    # getting top-k
    queries_topk = np.zeros((query_vecs.shape[0], index_vecs.shape[0]), dtype=int)
    for i in range(query_vecs.shape[0]):
        sims = sim_matrix[i]
        queries_topk[i] = sims.argsort()
    return queries_topk[:, :k]
