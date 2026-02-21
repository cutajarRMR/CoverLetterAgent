"""
ChromaDB memory management for embeddings.
https://www.trychroma.com/
Contents:
- get_collection: Retrieve or create a ChromaDB collection.
- get_embedding: Get embedding for a given text.
- add_to_memery: Add documents and their embeddings to the collection.
- query_bank_embeddings: Query the embeddings collection with distance thresholding.
"""


# Embed occupations data
import chromadb
from typing import List, Dict

def get_collection(embeddings, collection_name: str, path: str):
    """
    Retrieves or creates a ChromaDB collection.

    Args:
        embeddings (LLM): Embeddings model.
        collection_name (str): Name of the collection.
        path (str): Path to the persistent ChromaDB client.
    """
    client = chromadb.PersistentClient(
        path=path
    )
    model_name = getattr(embeddings, 'model', getattr(embeddings, 'model_id', 'unknown'))
    
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": model_name}
    )
    return collection, client


def get_embedding(embeddings, text: str):
    """
    Gets the embedding for a given text.

    Args:
        embeddings (LLM): Embeddings model.
        text (str): Text to embed.

    Returns:
        _type_: Embedding vector.
    """
    query_embedding = embeddings.embed_query(text)
    return query_embedding

def add_to_memery(embeddings_ai, collection_name: str, path: str, docs: list, ids: list):
    """
    Adds documents and their embeddings to the ChromaDB collection.

    Args:
        embeddings_ai (LLM): Embeddings model.
        collection_name (str): Name of the collection.
        path (str): Path to the persistent ChromaDB client.
        docs (list): _documents to add.
        ids (list): IDs for the documents.

    Returns:
        _type_: _description_
    """
    collection, client = get_collection(embeddings_ai, collection_name, path)
    embeddings = [get_embedding(embeddings_ai, i) for i in docs]
    collection.add(
                ids=ids,
        documents=docs,
        embeddings=embeddings
    )
    return f"Added {len(ids)} items to collection {collection_name}."



def query_bank_embeddings(
    embeddings,
    query: str, 
    collection_name: str, 
    path: str,
    n_results: int = 2, 
    distance_threshold: float = 0.6,  # adjust based on your embedding distance metric
    verbose = False
) -> List[Dict[str, float]]:
    """
    Queries the embeddings collection and returns IDs and distances,
    only if they are below a distance threshold.

    Args:
        embeddings (LLM): Embeddings model.
        query (str): The text query to search.
        collection_name (str): Name of the collection to query.
        path (str): Path to the persistent ChromaDB client.
        n_results (int, optional): Number of results to return. Defaults to 2.
        distance_threshold (float, optional): Max distance to consider a match. Defaults to 0.6.
        verbose (bool, optional): Whether to print debug info. Defaults to False.
        
    Returns:
        List[Dict[str, float]]: Each dict contains 'text_id' (ID), 'text', and 'distance'.
    """
    
    # Get the collection
    collection, client = get_collection(embeddings, collection_name, path)
    
    if verbose:
        print(f"Collection {collection.name} length: {collection.count()}")
        print(f"Query: {query}")
    
    # Get the embedding and query
    query_emb = get_embedding(embeddings, query)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=n_results
    )
    
    if verbose:
        print(f"Returned IDs: {results['ids'][0]}")
    
    # Filter by distance threshold
    output = []
    for i, _id in enumerate(results['ids'][0]):
        distance = results['distances'][0][i]
        if distance <= distance_threshold:
            output.append({
                'text_id': _id,
                'text': results['documents'][0][i],
                'distance': distance
            })
    
    return output
