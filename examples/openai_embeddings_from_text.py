from pyghostdb.ghost_storage import GhostStorage
from openai.api_resources.embedding import Embedding
import numpy as np


def calculate_embeddings_from_texts(texts: list[str]):
    # replace newlines, which can negatively affect performance.
    texts = [t.replace("\n", " ") for t in texts]
    # Call the OpenAI Embedding API in parallel for each document
    return [
        result["embedding"]
        for result in Embedding.create(
            input=texts,
            engine='text-embedding-ada-002',
        )["data"]
    ]


if __name__ == "__main__":
    # Initialize GhostStorage with default settings
    ghost_storage = GhostStorage()

    # Prepare some texts
    texts = ["Hello world!", "I love to code in Python", "Artificial Intelligence is fascinating"]

    # Calculate embeddings for the texts
    embeddings = calculate_embeddings_from_texts(texts)

    # Add texts and their embeddings to the storage
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        ghost_storage.upsert(i, text, embedding)

    # Search for the nearest neighbor of a query embedding
    query_embedding = np.random.rand(1536)
    result = ghost_storage.search(query_embedding, k=1)

    # Get the text and embedding of the nearest neighbor
    _, nearest_text, nearest_embedding = result[0]
    print(f"Nearest text: {nearest_text}, Nearest embedding: {nearest_embedding}")

    # Persist the index and text storage to disk
    ghost_storage.persist()

    # Load the index and text storage from disk
    ghost_storage.load()
