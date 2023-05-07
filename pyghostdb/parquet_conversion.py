import duckdb
import pyarrow as pa
import pyarrow.parquet


def write_to_parquet(text_ids: list[int], texts: list[str], embeddings: list[list[float]], output_file: str):
    schema = pa.schema([
        ('id_', pa.int32()),
        ('text', pa.string()),
        ('embedding', pa.list_(pa.float32())),
    ])
    data = {
        'id_': text_ids,
        'text': texts,
        'embedding': embeddings,
    }
    # Create Arrow arrays from the dictionary
    arrays = [pa.array(data[column], type=field_type) for column, field_type in zip(data.keys(), schema.types)]

    # Create an Arrow RecordBatch
    record_batch = pa.RecordBatch.from_arrays(arrays, schema=schema)

    # Convert the RecordBatch to an Arrow Table
    table = pa.Table.from_batches([record_batch])

    # Save the Table as a Parquet file
    pa.parquet.write_table(table, output_file)


def from_parquet_to_duckdb(parquet_file: str, duckdb_file: str, table_name: str):
    # Connect to DuckDB (this will create a new file if it doesn't exist)
    conn = duckdb.connect(duckdb_file)

    conn.execute(f''' DROP TABLE IF EXISTS {table_name} ''')
    conn.close()

    conn = duckdb.connect(duckdb_file)
    # Create a table with the appropriate schema
    conn.execute(f'''
        CREATE TABLE IF NOT EXISTS {table_name}  (id_ INTEGER PRIMARY KEY, text TEXT, embedding DOUBLE[]);
    ''')

    # Import the Parquet file into the table
    conn.execute(f'''
        COPY {table_name} FROM '{parquet_file}' (FORMAT 'parquet');
    ''')

    # Close the connection
    conn.close()


# (id_, text, vector_data)


if __name__ == "__main__":
    # Define your dictionary
    data = {
        'text': ['sample text 1', 'sample text 2'],
        'embedding': [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        'id_': [1, 2]
    }

    # Create an Arrow schema
    schema = pa.schema([
        ('id_', pa.int32()),
        ('text', pa.string()),
        ('embedding', pa.list_(pa.float64())),
    ])

    write_to_parquet(data['id_'], data['text'], data['embedding'], "test.parquet")

    from_parquet_to_duckdb("test.parquet", "my_database.duckdb", "my_table3355")
    # print the duckdb table
    conn = duckdb.connect("my_database.duckdb")
    print(conn.execute("SELECT * FROM my_table3355").fetchall())
    conn.close()
