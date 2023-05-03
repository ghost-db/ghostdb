import hnswlib
import numpy as np


class HNSWIndex:
    def __init__(self, dim, max_elements, ef=200, M=16):
        """
        :param dim: embedding dimension
        :param max_elements: maximum number of elements in the index
        :param ef: the size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more accurate but slower search. ef cannot be set lower than the number of queried nearest neighbors k. The value ef of can be anything between k and the size of the dataset.

        :param M: the number of bi-directional links created for every new element during construction.
        Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic dimensionality and/or high recall, while low M work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter also determines the algorithm's memory consumption, which is roughly M * 8-10 bytes per stored element.

        """
        self.dim = dim
        self.max_elements = max_elements
        self.ef = ef
        self.M = M
        self.index = None

    def init_index(self):
        """
        Initialize an empty hnswlib index
        """
        self.index = hnswlib.Index(space='cosine', dim=self.dim)
        self.index.init_index(max_elements=self.max_elements, ef_construction=self.ef, M=self.M)

    def build_index(self, embedding_vectors):
        """
        Build an hnswlib index from a set of embedding vectors
        """
        assert embedding_vectors.shape[1] == self.dim

        num_elements, dim = embedding_vectors.shape
        self.index = hnswlib.Index(space='cosine', dim=dim)  # possible options are l2, cosine or ip
        self.index.init_index(max_elements=num_elements, ef_construction=self.ef, M=self.M)
        self.index.add_items(embedding_vectors)
        self.index.set_ef(self.ef)  # ef should always be > k

    def add_items(self, embedding_vectors, ids=None):
        """
        Add new embedding vectors to the index
        """
        # assert embedding_vectors.shape[1] == self.dim

        self.index.add_items(embedding_vectors, ids=ids)

    def knn_query(self, embedding_vectors, k=1):
        """
        Query the index for the k nearest neighbors of the given embedding vectors
        """
        # check if 2d or 1d array
        if len(embedding_vectors.shape) == 1:
            embedding_vectors = np.expand_dims(embedding_vectors, axis=0)
        if embedding_vectors.shape[1] != self.dim:
            raise ValueError("embedding_vectors must be of shape (num_vectors, {})".format(self.dim))
        if k > self.ef:
            raise ValueError("k must be less than or equal to ef")

        labels, distances = self.index.knn_query(embedding_vectors, k=k)
        return labels, distances

    def save_to_file(self, path):
        """
        Save the index to disk
        """
        self.index.save_index(path)

    def load_from_file(self, path):
        """
        Load the index from disk
        """
        self.index.load_index(path)




