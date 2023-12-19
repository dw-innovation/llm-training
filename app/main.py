import os
from dotenv import load_dotenv

load_dotenv()
# comment the below lines if you need to use gpu 0.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv('GPU_DEVICE')

import torch
import re
import evaluate
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from app.tasks import TASKS
from app.models import MODELS
from typing import Dict
from tqdm import tqdm
from app.util import set_random_seed
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Text2TextGenerationPipeline)
from loguru import logger
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel

logger.add(f"logs/{__name__}.log", rotation="500 MB")


def train(params: Dict):
    task = params['task']
    max_length = params['max_length']
    cuda_device = params['cuda_device']
    model_type = params['model_type']
    pretrained_model = params['pretrained_model']
    train_file_path = params['train_file_path']
    val_file_path = params['val_file_path']
    debug = params['debug']
    model_output_path = params['model_output_path']
    eval_metric = params['eval_metric']
    epochs = params['epochs']
    learning_rate = params['learning_rate']

    task_func = TASKS[task]
    logger.info(f"Training a model for the task {task}.")
    logger.info(f"Max length is {max_length}")

    logger.info(f"Available devices are {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(torch.cuda.get_device_properties(i).name)

    torch.cuda.set_device(cuda_device)

    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

    logger.info(f"Selected device is {device}.")

    new_words = None
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    if 't5' in model_type:
        new_words = ['{', '}']
        logger.info("Adding the missing tokens.")
        tokenizer.add_tokens(new_words)

    # add metric func
    rouge_score = evaluate.load("rouge")
    logger.info("Added Rouge metric.")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the preds as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(pred.strip()) for pred in decoded_preds]

        decoded_labels = ["\n".join(label.strip()) for label in decoded_labels]
        # Compute ROUGscores
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    # dataset loading
    train_ds = task_func(dataset_path=train_file_path, tokenizer=tokenizer, max_length=max_length, debug=debug)
    val_ds = task_func(dataset_path=val_file_path, tokenizer=tokenizer, max_length=max_length, debug=debug)

    # declare model
    MODEL = MODELS[model_type]()
    model = MODEL.load(pretrained_model)

    if new_words:
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    # define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    # add LoRa adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # declare data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100,
                                           pad_to_multiple_of=8)

    # declare training arguments
    training_args = MODEL.training_args(
        output_dir=model_output_path,
        evaluation_strategy="epoch",
        do_eval=True,
        save_strategy="epoch",
        learning_rate=learning_rate,
        warmup_steps=int(len(train_ds) / 8),
        weight_decay=0.01,
        save_total_limit=1,
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=max_length,
        greater_is_better=True,
        auto_find_batch_size=True,
        metric_for_best_model=eval_metric,
        load_best_model_at_end=True
    )

    # declare trainer
    trainer = MODEL.trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    logger.info(f"Training of {model_type} started.")

    with torch.cuda.device(cuda_device):
        trainer.train()

    # Save our LoRA model & tokenizer results
    peft_model_id = model_output_path
    trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    logger.info(f"Training of {model_type} completed. The model is saved to {model_output_path}.")

    # remove the model
    del model
    del trainer

    if str(device) != "cpu":
        torch.cuda.empty_cache()


def test(params):
    task = params['task']
    max_length = params['max_length']
    cuda_device = params['cuda_device']
    model_type = params['model_type']
    test_file_path = params['test_file_path']
    model_output_path = params['model_output_path']
    result_file_path = params['result_file_path']
    debug = params['debug']

    peft_model_id = model_output_path
    config = PeftConfig.from_pretrained(model_output_path)

    # load base LLM model and tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    if 't5' in model_type:
        new_words = ['{', '}']
        logger.info("Adding the missing tokens.")
        tokenizer.add_tokens(new_words)

    model = PeftModel.from_pretrained(model, peft_model_id, device_map={"": 0})
    model = model.merge_and_unload()
    model.eval()

    task_func = TASKS[task]
    test_ds = task_func(dataset_path=test_file_path, tokenizer=tokenizer, max_length=max_length, debug=debug, test=True)

    test_files = test_ds['sentence'].tolist()
    device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")
    pipeline = Text2TextGenerationPipeline(model=model, batch_size=16,
                                           tokenizer=tokenizer,
                                           device=device,  # model.device,
                                           clean_up_tokenization_spaces=True)
    logger.info('Getting predictions...')
    generated_texts = pipeline(test_files, do_sample=False, max_length=max_length, pad_token_id=tokenizer.pad_token_id)

    logger.info('Predictions is done.')

    predictions = []
    for test_inst, generated_text in tqdm(zip(test_files, generated_texts), total=len(test_files)):
        predictions.append({
            "sentence": test_inst,
            "generated_sentence": generated_text['generated_text'],
            "expected_sentence": test_ds.loc[test_ds['sentence'] == test_inst]['query'].values[0] if "query" in test_ds else None
        })

    predictions = pd.DataFrame(predictions)
    predictions.to_csv(result_file_path, sep='\t')


if __name__ == '__main__':
    parser = ArgumentParser()

    # general settings
    parser.add_argument("--train_file_path")
    parser.add_argument("--val_file_path")
    parser.add_argument("--test_file_path")
    parser.add_argument('--model_output_path')
    parser.add_argument("--result_file_path")
    parser.add_argument('--random_seed', type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")

    # task settings
    parser.add_argument("--task")

    # model settings
    parser.add_argument("--pretrained_model")
    parser.add_argument("--model_type")
    parser.add_argument("--max_length", type=int)

    # training settings
    parser.add_argument("--cuda_device", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--eval_metric", type=str)

    args = parser.parse_args()
    params = vars(args)

    set_random_seed(args.random_seed)

    if args.train:
        train(params)

    if args.test:
        test(params)
