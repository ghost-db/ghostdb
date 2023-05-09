# GhostDB

GhostDB is a Python package that provides a fast and efficient way to store and search for embeddings using HNSW (Hierarchical Navigable Small World) indexing from the hnswlib library. It is designed for applications that require fast nearest neighbor search, such as natural language processing, image recognition, and recommendation systems.

## Features

- Fast approximate nearest neighbor search using HNSW indexing
- Support for high-dimensional data
- Persistence of index and text data on disk
- Easy-to-use API for adding, searching, and managing embeddings
- Support for batch operations

## Installation

```bash
pip install pyghostdb
```

## Usage

Here is an example of how to use GhostDB:

```python
import numpy as np
from pyghostdb.ghost_storage import GhostStorage

# Initialize GhostStorage with default settings
ghost_storage = GhostStorage()

# Add an embedding to the storage
text_id = 1
text = "Sample text"
embedding = np.random.rand(1536)
ghost_storage.add(text_id, text, embedding)

# Search for the nearest neighbor of a query embedding
query_embedding = np.random.rand(1536)
result = ghost_storage.search(query_embedding, k=1)

# Get the text and embedding of the nearest neighbor
nearest_text, nearest_embedding = result[0]
print(f"Nearest text: {nearest_text}, Nearest embedding: {nearest_embedding}")

# Persist the index and text storage to disk
ghost_storage.persist()

# Load the index and text storage from disk
ghost_storage.load()
```

## API Reference

### `class GhostStorage`

The main class for storing and searching embeddings.

#### `__init__(self, dim=1536, max_elements=10**5, persist_dir="ghost_dir")`

Initialize the GhostStorage instance.

- `dim`: The dimension of the embeddings (default: 1536)
- `max_elements`: The maximum number of elements that can be stored in the index (default: 10^5)
- `persist_dir`: The directory where the index and text storage will be persisted (default: "ghost_dir")

#### `add(self, text_id, text, embedding)`

Add a single text and its embedding to the storage.

- `text_id`: The unique identifier of the text
- `text`: The text associated with the embedding
- `embedding`: The embedding as a numpy array or list

#### `add_multiple(self, ids: list[int], texts: list[str], embeddings: np.ndarray)`

Add multiple texts and their embeddings to the storage.

- `ids`: A list of unique identifiers for the texts
- `texts`: A list of texts associated with the embeddings
- `embeddings`: A numpy array containing the embeddings

#### `search(self, embedding, k=1)`

Search for the k nearest neighbors of a query embedding.

- `embedding`: The query embedding as a numpy array or list
- `k`: The number of nearest neighbors to search for (default: 1)

Returns a list of tuples containing the text and its embedding for each nearest neighbor.

#### `clear(self)`

Clear the storage and remove the index and text storage from disk.

#### `persist(self)`

Persist the index and text storage to disk.

#### `load(self)`

Load the index and text storage from disk.