import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Learning Vector databases: ChromaDB
    """)
    return


@app.cell
def _():
    import marimo as mo
    import chromadb
    from sentence_transformers import SentenceTransformer
    from sklearn.datasets import fetch_20newsgroups
    import numpy as np
    import pandas as pd
    import re
    import os
    import sqlite3

    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    return (
        SentenceTransformer,
        chromadb,
        fetch_20newsgroups,
        mo,
        np,
        os,
        re,
        sqlite3,
    )


@app.cell
def _(fetch_20newsgroups, np):
    # Load dataset
    categories = ['comp.graphics', 'comp.sys.mac.hardware']
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )

    # Sample subset
    n_samples = 50
    np.random.seed(42)
    indices = np.random.choice(len(newsgroups.data), n_samples, replace=False)

    sample_data = [newsgroups.data[i] for i in indices]
    sample_targets = newsgroups.target[indices]

    print(f"Loaded {len(sample_data)} documents")
    print(f"Categories: {categories}")
    print(f"Sample: {sample_data[0][:200]}...")
    return newsgroups, sample_data, sample_targets


@app.cell
def _(newsgroups, re, sample_data, sample_targets):
    # Clean text
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    cleaned_data = []
    metadata = []

    for i, text in enumerate(sample_data):
        cleaned = clean_text(text)
        if len(cleaned) > 50:
            cleaned_data.append(cleaned)
            metadata.append({
                'category': newsgroups.target_names[sample_targets[i]],
                'doc_id': i,
                'length': len(cleaned)
            })

    print(f"Cleaned {len(cleaned_data)} documents")
    return cleaned_data, metadata


@app.cell
def _(SentenceTransformer, cleaned_data):
    # Generate embeddings
    embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    embeddings = embedder.encode(cleaned_data)

    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    return embedder, embeddings


@app.cell
def _(chromadb, cleaned_data, embeddings, metadata):
    # Setup ChromaDB
    client = chromadb.PersistentClient(path="./chroma_db")
    collection_name = "newsgroups"

    try:
        client.delete_collection(collection_name)
    except:
        pass

    collection = client.create_collection(collection_name)

    # Add documents
    ids = [f"doc_{i}" for i in range(len(cleaned_data))]
    collection.add(
        embeddings=embeddings.tolist(),
        documents=cleaned_data,
        metadatas=metadata,
        ids=ids
    )

    print(f"Added {collection.count()} documents to ChromaDB")
    return (collection,)


@app.cell
def _(collection, os, sqlite3):
    # Explore ChromaDB internals
    print("ChromaDB Storage Exploration\n")

    # Collection metadata
    print("Collection Info:")
    print(f"Name: {collection.name}")
    print(f"Count: {collection.count()}")
    print(f"ID: {collection.id}")

    # Sample documents with all data
    print(f"\nSample Documents:")
    sample_docs = collection.get(
        ids=["doc_0", "doc_1"], 
        include=['embeddings', 'documents', 'metadatas']
    )

    for j, (docid, docname, docmeta, docemb) in enumerate(zip(
        sample_docs['ids'], sample_docs['documents'], sample_docs['metadatas'], sample_docs['embeddings']
    )):
        print(f"\n{j+1}. ID: {docid}")
        print(f"Category: {docmeta['category']}")
        print(f"Doc length: {docmeta['length']} chars")
        print(f"Embedding: {len(docemb)} dims, first 5: {docemb[:5]}")
        print(f"Text: {docname[:100]}...")

    # Check disk files
    print(f"\nFiles on disk:")
    if os.path.exists("./chroma_db"):
        for root, dirs, files in os.walk("./chroma_db"):
            for file in files:
                filepath = os.path.join(root, file)
                size = os.path.getsize(filepath)
                print(f"{filepath} ({size:,} bytes)")

    try:
        db_files = [f for f in os.listdir("./chroma_db") if f.endswith('.sqlite3')]
        if db_files:
            print(f"\nSQLite tables in {db_files[0]}:")
            conn = sqlite3.connect(f"./chroma_db/{db_files[0]}")
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            for table in tables:
                print(f"- {table[0]}")

            try:
                cursor = conn.execute("SELECT * FROM collections LIMIT 2")
                print(f"\nSample from collections table:")
                for row in cursor.fetchall():
                    print(f"{row}")
            except:
                pass

            conn.close()
    except Exception as e:
        print(f"SQLite read error: {e}")
    return


@app.cell
def _(collection, embedder):
    # Test search
    def search_database(query, n_results=3):
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

        print(f"Query: '{query}'")
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            similarity = 1 / (1 + dist)
            print(f"{i+1}. [{meta['category']}] (Score: {similarity:.3f})")
            print(f"   {doc[:150]}...")
            print()

        return results

    # Run test searches
    search_database("How do I improve graphics performance?")
    search_database("Mac hardware recommendations")
    return


if __name__ == "__main__":
    app.run()
