import pandas as pd
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
from loguru import logger
from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

load_dotenv()

index_name = "training-embeddings"
model = SentenceTransformer(os.getenv('SENT_TRANSFORMER'))


def index_from_file(fpath, batch_size):
    es = Elasticsearch(
        os.getenv("SEARCH_ENGINE_HOST"),  # Elasticsearch endpoint

    )
    mappings = {
        "properties": {
            "embeddings": {
                "type": "dense_vector",
                "dims": 768,
                "index": "true",
                "similarity": "cosine"
            }
        }
    }

    # create index
    es.indices.create(
        index=index_name,
        mappings=mappings,
        ignore=[400, 404]
    )
    #

    data = pd.read_csv(fpath, sep='\t')
    actions = []
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentence = row['sentence']
        query = row['query']

        action = {"index": {"_index": index_name}}

        doc = {
            "sentence": sentence,
            "query": query,
            "embeddings": model.encode(sentence)
        }

        actions.append(action)
        actions.append(doc)

        if len(actions) > batch_size:
            es.bulk(index=index_name, operations=actions)
            logger.info(f"{len(actions)} samples indexed.")
            actions.clear()

    if len(actions) > 0:
        es.bulk(index=index_name, operations=actions)

    result = es.count(index=index_name)
    logger.info(f"{result.body['count']} tags indexed.")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--training_fname', type=str, help='Training sets')
    parser.add_argument('--batch_size', type=int, help='Batch size', default=1000)

    args = parser.parse_args()

    index_from_file(args.training_fname, args.batch_size)
