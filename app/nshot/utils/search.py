import os
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from loguru import logger

load_dotenv()

index_name = "training-embeddings"
model = SentenceTransformer(os.getenv('SENT_TRANSFORMER'))

client = Elasticsearch(
    os.getenv("SEARCH_ENGINE_HOST"),  # Elasticsearch endpoint

)


def construct_knn_query(query_vector, num_candidates: int = 100, k: int = 10):
    return {
        "field": "embeddings",
        "query_vector": query_vector,
        "k": k,
        "num_candidates": num_candidates
    }


def search_similar_sentence(sentence, few_shot):
    query_vector = model.encode(sentence)
    resp = client.knn_search(index=index_name,
                             knn=construct_knn_query(query_vector),
                             source=["sentence", "query"])

    results = resp['hits']['hits'][:few_shot]

    search_results = []

    for result in results:
        search_results.append({
            'sentence': result['_source']['sentence'],
            "query": result['_source']['query'],
            "score": result['_score'],
        })

        return search_results
