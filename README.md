## Usage

1. Import the `GhostStorage` class from `ghost_storage.py`.

```python
from ghost_storage import GhostStorage
```

2. Initialize the `GhostStorage` object with desired parameters (optional).

```python
ghost_storage = GhostStorage(dim=1024, max_elements=10**5)
```

3. Add a new text and its corresponding embedding.

```python
text_id = 1
text = "Some text"
embedding = np.array([...])  # numpy array with the embedding

ghost_storage.add(text_id, text, embedding)
```

4. Search for similar texts using an embedding.

```python
query_embedding = np.array([...])  # numpy array with the query embedding
k = 5  # number of nearest neighbors to search for

results = ghost_storage.search(query_embedding, k=k)
```

5. Clear the storage.

```python
ghost_storage.clear()
```
