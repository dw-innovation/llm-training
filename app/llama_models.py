import os
import json
from dotenv import load_dotenv
from app.nshot.utils import read_prompt_file

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
                          EarlyStoppingCallback, pipeline, TrainingArguments, BitsAndBytesConfig)
from loguru import logger
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType, PeftConfig, PeftModel
from datasets import Dataset

# reference: https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama_2/scripts/sft_llama2.py

class Llama2Model:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def train(self, train_ds, val_ds, params: Dict):
        # todo: rename the function name
        instruction_file = read_prompt_file(prompt_file=f"scripts/{params['task']}/prompt/zero_shot_cot_prompt.txt")

        if params['debug']:
            random_seed = params['random_seed']
            sample_ratio = params['sample_ratio']
            train_ds = train_ds.sample(int(len(train_ds)*sample_ratio), random_state=random_seed)
            val_ds = val_ds.sample(int(len(val_ds)*sample_ratio), random_state=random_seed)


        train_ds['sentence'] = train_ds['sentence'].apply(lambda x: x.lower())
        train_ds['instruction'] = train_ds.apply(lambda example: f"### Instruction: {instruction_file}\n ### Input: {example.sentence}\n ### Response: {example.query}", axis=1)

        val_ds['sentence'] = val_ds['sentence'].apply(lambda x: x.lower())
        val_ds['instruction'] = val_ds.apply(lambda example: f"### Instruction: {instruction_file}\n ### Input: {example.sentence}\n ### Response: {example.query}", axis=1)

        train_ds = Dataset.from_pandas(train_ds)
        val_ds = Dataset.from_pandas(val_ds)

        cuda_device = params['cuda_device']

        logger.info(f"Available devices are {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(torch.cuda.get_device_properties(i).name)

        torch.cuda.set_device(cuda_device)

        device = torch.device(f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu")

        logger.info(f"Selected device is {device}.")

        # define LoRA Config
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )


        # declare model
        print(f"pretrained model {params['pretrained_model']}")
        model = AutoModelForCausalLM.from_pretrained(params['pretrained_model'],
                                                     low_cpu_mem_usage=True,
                                                     trust_remote_code=True,
                                                     quantization_config=bnb_config,
                                                     return_dict=True,
                                                     torch_dtype=torch.float16,
                                                     device_map="auto",
                                                     )

        model.config.use_cache = False

        # recheck the following
        model.config.pretraining_tp = 1 


        model_output_path = params['model_output_path']
        learning_rate = params['learning_rate']
        epochs = params['epochs']
        max_length = params['max_length']

        # add metric func
        rouge_score = evaluate.load("rouge")
        logger.info("Added Rouge metric.")
        
        
        def preprocess_logits_for_metrics(logits, labels):
            # the actualy prediction is the first element of the tuple
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)


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


        # declare training arguments
        training_args = TrainingArguments(
            output_dir=params['model_output_path'],
            evaluation_strategy="steps",
            do_eval=True,
            save_strategy="steps",
            eval_steps=100,
            learning_rate=learning_rate,
            logging_strategy='steps', # log according to log_steps
            warmup_steps=int(len(train_ds) / 8),
            weight_decay=0.01,
            save_total_limit=1,
            num_train_epochs=epochs,
            auto_find_batch_size=True,
            metric_for_best_model=params['eval_metric'],
            greater_is_better=True,
            # metric_for_best_model='eval_loss',
            load_best_model_at_end=True
        )

        # declare trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            dataset_text_field="instruction",
            peft_config=lora_config,
            tokenizer=self.tokenizer,
            max_seq_length= max_length,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics = preprocess_logits_for_metrics,
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


    @torch.inference_mode()
    def test(self, test_ds, params):
        peft_model_id = params['model_output_path']
        config = PeftConfig.from_pretrained(params['model_output_path'])

        # load base LLM model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto')
        model = model.merge_and_unload()

        test_sentences = test_ds['sentence'].tolist()

        device = torch.device(params['cuda_device'] if torch.cuda.is_available() else "cpu")

        model.eval()

        model.to(device)

        instruction_text = read_prompt_file(prompt_file=f"scripts/{params['task']}/prompt/zero_shot_cot_prompt.txt")
        
        with open(params['result_file_path'], 'w') as outfile:
            for sentence in tqdm(test_sentences, total=len(test_sentences)):
                sentence = sentence.lower()
                prompt = f"### Instruction: {instruction_text}\n ### Input: {sentence}\n ### Response:\n "
                input_ids = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.cuda()

                outputs = model.generate(input_ids=input_ids, max_new_tokens=params['max_length'], do_sample=True, top_p=0.9,temperature=0.5)
                generated_instruction =tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

                processed_data = {
                    "sentence": sentence,
                    "model_result": generated_instruction
                }

                json.dump(processed_data, outfile)
                outfile.write('\n')