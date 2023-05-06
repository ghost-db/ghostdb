from hnswlib_index import HNSWIndex
import numpy as np
from text_storage import TextStorage
from ghost_storage import GhostStorage

def test_hnsw():
    n_elements = 100
    index = HNSWIndex(dim=1024, max_elements=n_elements)
    index.build_index(np.float32(np.random.random((n_elements, 1024))))
    test_data = np.float32(np.random.random((100, 1024)))
    labels1, distances = index.knn_query(test_data, k=1)
    index.add_items


def test_small_vectors():
    n_elements = 100
    index = HNSWIndex(dim=3, max_elements=n_elements)
    index.build_index(np.float32(np.random.random((n_elements, 3))))
    test_data = np.float32(np.random.random((100, 3)))
    labels1, distances = index.knn_query(test_data, k=1)
    print(distances)


def test_small_vectors2():
    n_elements = 100
    index = HNSWIndex(dim=3, max_elements=n_elements)
    index.init_index()
    index.add_items(np.float32(np.random.random((n_elements, 3))), list(map(lambda x: str(x), range(n_elements))))
    test_data = np.float32(np.random.random((5, 3)))
    labels1, distances = index.knn_query(test_data, k=3)
    print(labels1)


def test_text_storage():
    embedding = [1.23, 4.56, 7.89]
    text = "hello world"
    id_ = "123"
    text_storage = TextStorage(db_path="trademark.db")
    text_storage.add(id_=id_, text=text, embedding=embedding)
    id_, text, vector_data = text_storage.load(id_=id_)
    text_storage.remove_db()
    print(id_, text, vector_data)


def test_ghostdb():
    gs = GhostStorage(dim=3, max_elements=100)
    gs.add(1, 'hello world', np.random.random((3, 1)))
    gs.add(2, 'hello world 2', np.random.random((3, 1)))
    gs.add(3, 'hello world 3', np.random.random((3, 1)))
    gs.add(4, 'hello world 4', np.random.random((3, 1)))
    gs.add(5, 'hello world 5', np.random.random((3, 1)))
    gs.add(5, 'hello world 6', np.random.random((3, 1)))

    print(gs.search(np.array([1, 2, 3]), k=1))

def test_add_multiple():
    gs = GhostStorage(dim=3, max_elements=100)
    gs.add_multiple([1, 2, 3, 4, 5, 6], ['hello world', 'hello world 2', 'hello world 3', 'hello world 4', 'hello world 5', 'hello world 6'], np.random.random((6, 3)))
    print(gs.search(np.array([1, 2, 3]), k=1))


if __name__ == '__main__':
    test_hnsw()
    test_small_vectors2()
    test_text_storage()
    test_ghostdb()
    test_add_multiple()

