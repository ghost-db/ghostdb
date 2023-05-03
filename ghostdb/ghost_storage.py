import numpy as np

from hnswlib_index import HNSWIndex
from text_storage import TextStorage


class GhostStorage:
    hnsw_index = None
    text_storage_db = None

    def __init__(self, dim=1024, max_elements=10**5):
        self.hnsw_index = HNSWIndex(dim=dim, max_elements=max_elements, ef=200, M=16)
        self.hnsw_index.init_index()
        self.text_storage_db = TextStorage("text_storage.db")

    def add(self, text_id, text, embedding):
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == self.hnsw_index.dim
        assert self.hnsw_index is not None
        # convert numpy array to list of floats
        embedding = [float(x) for x in list(embedding.reshape(-1))]
        self.text_storage_db.add(text_id, text, embedding)
        self.hnsw_index.add_items(embedding, [text_id])

    def search(self, embedding, k=1):
        ids, distance = self.hnsw_index.knn_query(embedding, k=k)
        return [self.text_storage_db.get(id_) for id_ in ids]

    def clear(self):
        self.hnsw_index = None
        self.text_storage_db.remove_db()



