import numpy as np

from pyghostdb.hnswlib_index import HNSWIndex
from pyghostdb.text_storage import TextStorage
import os


class GhostStorage:
    hnsw_index = None
    text_storage_db = None

    def __init__(self, dim=1024, max_elements=10**5, persist_dir="ghost_dir"):
        self.hnsw_index = HNSWIndex(dim=dim, max_elements=max_elements, ef=200, M=16)
        self.hnsw_index.init_index()
        self.persist_dir = persist_dir
        if not os.path.exists(self.persist_dir):
            os.makedirs(self.persist_dir)
        self.text_storage_db = TextStorage(os.path.join(self.persist_dir, "text_storage.db"))

    def add(self, text_id, text, embedding):
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == self.hnsw_index.dim
        assert self.hnsw_index is not None
        # convert numpy array to list of floats
        embedding = [float(x) for x in list(embedding.reshape(-1))]
        self.text_storage_db.add(text_id, text, embedding)
        self.hnsw_index.add_items(embedding, [text_id])

    def add_multiple(self, ids: list[int], texts: list[str], embeddings: np.ndarray):
        # convert numpy array to list of floats
        self.hnsw_index.add_items(embeddings, ids)
        embeddings = [list(embedding) for embedding in embeddings]
        self.text_storage_db.add_multiple(ids, texts, embeddings)

    def search(self, embedding, k=1):
        ids, distance = self.hnsw_index.knn_query(embedding, k=k)
        # convert ids to list of ints
        ids = [int(id_) for id_ in ids.flatten().tolist()]
        return [self.text_storage_db.get(id_) for id_ in ids]

    def clear(self):
        self.hnsw_index = None
        self.text_storage_db.remove_db()

    def hnws_index_filepath(self):
        return os.path.join(self.persist_dir, "index.ghostdb")

    def persist(self):
        self.hnsw_index.save_to_file(self.hnws_index_filepath())

    def load(self):
        self.hnsw_index.load_from_file(self.hnws_index_filepath())
