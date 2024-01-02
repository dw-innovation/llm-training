import os
import json
from dotenv import load_dotenv

load_dotenv()
# comment the below lines if you need to use gpu 0.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv('GPU_DEVICE')

import torch
import evaluate
import numpy as np
from typing import Dict
from tqdm import tqdm
from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq,
                          EarlyStoppingCallback, Text2TextGenerationPipeline,TrainingArguments)
from loguru import logger
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel


class Llama2Model:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def train(self, train_ds, val_ds, params: Dict):
        cuda_device = params['cuda_device']

        logger.info(f"Available devices are {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(torch.cuda.get_device_properties(i).name)

        torch.cuda.set_device(cuda_device)

        device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

        logger.info(f"Selected device is {device}.")

        # add metric func
        rouge_score = evaluate.load("rouge")
        logger.info("Added Rouge metric.")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            if isinstance(preds, tuple):
                preds = preds[0]

            # Replace -100 in the preds as we can't decode them
            preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)

            # Decode generated summaries into text
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            # Replace -100 in the labels as we can't decode them
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            # Decode reference summaries into text
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
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

        # declare model
        print(f"pretrained model {params['pretrained_model']}")
        model = AutoModelForCausalLM.from_pretrained(params['pretrained_model'],
                                                     low_cpu_mem_usage=True,
                                                     trust_remote_code=True,
                                                     return_dict=True,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto",
                                                     )

        model.to(device)

        # define LoRA Config
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )

        # add LoRa adaptor
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # declare data collator
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=model, label_pad_token_id=-100,
                                               pad_to_multiple_of=8)

        model_output_path = params['model_output_path']
        learning_rate = params['learning_rate']
        epochs = params['epochs']
        max_length = params['max_length']

        # declare training arguments
        training_args = TrainingArguments(
            output_dir=params['model_output_path'],
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
            metric_for_best_model=params['eval_metric'],
            load_best_model_at_end=True
        )

        # declare trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )

        with torch.cuda.device(cuda_device):
            trainer.train()

        # Save our LoRA model & tokenizer results
        peft_model_id = model_output_path
        trainer.model.save_pretrained(peft_model_id)
        self.tokenizer.save_pretrained(peft_model_id)

        # remove the model
        del model
        del trainer

        if str(device) != "cpu":
            torch.cuda.empty_cache()

    def test(self, test_sentences, params):
        peft_model_id = params['model_output_path']
        config = PeftConfig.from_pretrained(params['model_output_path'])

        # load base LLM model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        model = PeftModel.from_pretrained(model, peft_model_id, device_map={"": 0})
        model = model.merge_and_unload()
        model.eval()

        device = torch.device(f"cuda:{params['cuda_device']}" if torch.cuda.is_available() else "cpu")
        pipeline = Text2TextGenerationPipeline(model=model, batch_size=16,
                                               tokenizer=tokenizer,
                                               device=device,  # model.device,
                                               clean_up_tokenization_spaces=True)
        logger.info('Getting predictions...')
        generated_texts = pipeline(test_sentences, do_sample=False, max_length=params['max_length'],
                                   pad_token_id=self.tokenizer.pad_token_id)

        logger.info('Predictions is done.')

        with open(params['result_file_path'], 'w') as outfile:
            for test_inst, generated_text in tqdm(zip(test_sentences, generated_texts), total=len(test_sentences)):
                processed_data = {
                    "sentence": test_inst,
                    "model_result": generated_text['generated_text']
                }
                json.dump(processed_data, outfile)
                outfile.write('\n')
