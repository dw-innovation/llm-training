from typing import Dict, Any
from datasets import Dataset
import pandas as pd
import json


def preprocess_function(examples: Dict[str, Any], max_length: int, tokenizer: object, input_col: str, output_col: str,
                        output_type: str) -> Dict[str, Any]:
    """preprocess each row of op++ datasets by create input with this format
        input is text
        the labels will be the dictionary

    Args:
        examples (Dict[str, Any]): each row of datasets

    Returns:
        output from tokenizer
    """
    inputs = examples[input_col]
    if output_col:
        if output_type == 'json':
            targets = [json.dumps(example) for example in examples[output_col]]
        else:
            targets = examples[output_col]

        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True,
                                 return_tensors="np")
    else:
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True,
                                 return_tensors="np")
    return model_inputs

def load_spot_dataset(dataset, tokenizer, max_length, test=False, output_type='yaml'):
    output_col = None

    dataset["sentence"] = dataset["sentence"].apply(lambda x: x.lower())

    if "query" in dataset:
        dataset["query"] = dataset["query"].apply(lambda x: x.lower())
        output_col = 'query'

    if test:
        return dataset[['sentence']]

    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.map(
        lambda x: preprocess_function(x, max_length=max_length, tokenizer=tokenizer, input_col="sentence",
                                      output_col=output_col, output_type=output_type), batched=True)
    return dataset


TASKS = {
    "spot": load_spot_dataset
}
