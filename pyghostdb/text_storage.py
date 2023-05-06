import duckdb
import os
from pyghostdb.parquet_conversion import write_to_parquet, from_parquet_to_duckdb


class TextStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def add(self, id_: str, text: str, embedding: list[float]):
        connection = duckdb.connect(self.db_path)
        cursor = connection.cursor()

        # Create a table with an ARRAY column to store the vector
        cursor.execute("CREATE TABLE IF NOT EXISTS vectors (id_ INTEGER PRIMARY KEY, text TEXT, embedding DOUBLE[])")

        # remove the old vector from the table if it exists
        cursor.execute("DELETE FROM vectors WHERE id_ = ?", (id_,))
        # Insert the vector into the table
        cursor.execute("""
            INSERT INTO vectors (id_, text, embedding) 
            VALUES (?, ?, ?) 
        """, (id_, text, embedding))

        connection.commit()

    def add_multiple(self, ids: list[int], texts: list[str], embeddings: list[list[float]]):
        # write to parquet
        parquet_filepath = f"{self.db_path}_tmp.parquet"
        write_to_parquet(ids, texts, embeddings, parquet_filepath)
        from_parquet_to_duckdb(parquet_filepath, self.db_path, "vectors")


    def get(self, id_):
        connection = duckdb.connect(self.db_path)
        cursor = connection.cursor()

        id_int = int(id_)
        # Load the vector from the table
        cursor.execute("SELECT id_, text, embedding FROM vectors WHERE id_ = ?", (id_int,))
        id_, text, vector_data = cursor.fetchone()

        return id_, text, vector_data

    def get_text(self, id_):
        connection = duckdb.connect(self.db_path)
        cursor = connection.cursor()

        id_int = int(id_)
        # Load the vector from the table
        cursor.execute("SELECT id_, text FROM vectors WHERE id_ = ?", (id_int,))
        text = cursor.fetchone()

        return text

    def load(self, id_: str):
        connection = duckdb.connect(self.db_path)
        cursor = connection.cursor()

        # Load the vector from the table
        cursor.execute("SELECT id_, text, embedding FROM vectors WHERE id_ = ?", (id_,))
        id_, text, vector_data = cursor.fetchone()

        return id_, text, vector_data

    def remove_db(self):
        os.remove(self.db_path)



