import os
import torch
import pandas as pd
import json
from enum import Enum
from tqdm import tqdm
from argparse import ArgumentParser
from app.nshot.json_extractor import json_extractor
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel
from typing import List, Optional
from app.nshot.utils import read_prompt_file
from sentence_transformers import SentenceTransformer
import yaml

load_dotenv()

class EntityType(Enum):
    NWR = 'nwr'
    CLUSTER = 'cluster'

class RelationType(Enum):
    DIST = 'dist'
    CONTAINS = 'contains'

class Area(BaseModel):
    type: str
    value: Optional[str]

class Property(BaseModel):
    name: str
    operator: str
    value: str

class Entity(BaseModel):
    name: str
    id: str
    type: EntityType
    properties: List[Property]

class Relation(BaseModel):
    source: str
    target: str
    type: RelationType
    value: Optional[str]

class Output(BaseModel):
    area: Area
    entities: List[Entity]
    relations: List[Relation]

def generate(client, query, prompt, response_prompt):
    logger.info(f"Prompting {query}")
    completion = client.beta.chat.completions.parse(
        model=os.getenv("CHATGPT_MODEL"),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.6,
        response_format=response_prompt
    )
    logger.info(f"chatgpt returned a result")
    raw_output = completion.choices[0].message.content
    # logger.info(f"the document is parsing")
    # parsed_output = json_extractor(raw_output)
    # logger.info(f"the document parsed")
    return raw_output

if __name__ == '__main__':
    if __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument("--prompt_file", type=str, required=True)
        parser.add_argument("--input_file", type=str, required=True)
        parser.add_argument("--result_file", type=str, required=True)
        parser.add_argument('--few_shot', type=int, default=-1)

        args = parser.parse_args()
        few_shot = args.few_shot

        embedder = None
        if few_shot!=-1:
            few_shot_data = pd.read_csv('tasks/spot/dev_v16_3-17_1-2.tsv', sep='\t')
            few_shot_data['sentence'] = few_shot_data['sentence'].apply(lambda x: x.lower())
            corpus = few_shot_data['sentence'].tolist()
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

        few_shot = args.few_shot

        sys_message = read_prompt_file(prompt_file=args.prompt_file)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], organization=os.environ["OPENAI_ORG"])

        with open(args.input_file, 'r') as input_file:
            sentences = input_file.readlines()
            sentences = sentences[1:-1]
            with open(args.result_file, 'w') as outfile:
                for sentence in tqdm(sentences, total=len(sentences)):
                    json_data = {"sentence": sentence}
                    sentence = sentence.rstrip().lower()

                    if few_shot!= -1:
                        query_embedding = embedder.encode(sentence, convert_to_tensor=True)
                        similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
                        scores, indices = torch.topk(similarity_scores, k=few_shot)

                        examples = ''
                        for score, idx in zip(scores, indices):
                            example = corpus[idx]
                            yaml_out_str = few_shot_data[few_shot_data['sentence'] == example]['query'].values[0]
                            # yaml_out = yaml.safe_load(yaml_out_str)
                            # print(yaml_out)
                            examples += '===Input===' + '\n' + example + '\n' + '===Output===' + '\n' + str(yaml_out_str)

                    sentence = examples + '\n' + '===Input==' + '\n' + sentence + '\n' + '===Output===' + '\n'

                    generated_text = generate(client=client, query=sentence, prompt=sys_message, response_prompt=Output)
                    # if isinstance(generated_text, tuple):
                    #     generated_text = generated_text
                    generated_text = generated_text.replace('```yaml', '').replace('```', '')
                    generated_text = generated_text.replace('*', '-').replace('+', '-')
                    json_data["model_result"] = generated_text
                    # json_data["parsed_result"] = json_extractor(generated_text)
                    json.dump(json_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
