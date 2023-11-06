from typing import Dict, Any
from datasets import Dataset
import pandas as pd
import json


def preprocess_function(examples: Dict[str, Any], max_length: int, tokenizer: object, input_col: str, output_col: str):
    """preprocess each row of op++ datasets by create input with this format
        input is text
        the labels will be the dictionary

    Args:
        examples (Dict[str, Any]): each row of datasets

    Returns:
        output from tokenizer
    """
    inputs = examples[input_col]
    targets = [json.dumps(example) for example in examples[output_col]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True, return_tensors="np")
    return model_inputs


def load_spot_dataset(dataset_path, tokenizer, max_length, debug, test=False):
    if debug:
        dataset = pd.read_csv(dataset_path, sep='\t')[:20]
    else:
        dataset = pd.read_csv(dataset_path, sep='\t')
    dataset["sentence"] = dataset["sentence"].apply(lambda x: x.lower())
    dataset["query"] = dataset["query"].apply(lambda x: x.lower())

    if test:
        return dataset[['sentence', 'query']]

    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(
        lambda x: preprocess_function(x, max_length=max_length, tokenizer=tokenizer, input_col="sentence",
                                      output_col="query"), batched=True)
    return dataset


TASKS = {
    "spot": load_spot_dataset
}
