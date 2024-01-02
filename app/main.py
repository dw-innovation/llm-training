import os
from dotenv import load_dotenv
from app.models import MODELS
from app.tasks import TASKS
from transformers import AutoTokenizer

load_dotenv()
# comment the below lines if you need to use gpu 0.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv('GPU_DEVICE')

from argparse import ArgumentParser
from app.util import set_random_seed
from loguru import logger

logger.add(f"logs/{__name__}.log", rotation="500 MB")

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

    print(f"pretrained model {args.pretrained_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model, trust_remote_code=True)

    if 't5' in args.model_type:
        new_words = ['{', '}']
        logger.info("Adding the missing tokens.")
        tokenizer.add_tokens(new_words)
    elif 'llama2' in args.model_type:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model = MODELS[args.model_type](tokenizer=tokenizer)
    task_func = TASKS[args.task]
    task = args.task

    if args.train:
        max_length = params['max_length']
        train_file_path = params['train_file_path']
        val_file_path = params['val_file_path']
        debug = params['debug']

        logger.info(f"Training a model for the task {task}.")
        logger.info(f"Max length is {max_length}")

        # dataset loading
        train_ds = task_func(dataset_path=train_file_path, tokenizer=tokenizer, max_length=max_length, debug=debug)
        val_ds = task_func(dataset_path=val_file_path, tokenizer=tokenizer, max_length=max_length, debug=debug)

        model.train(train_ds, val_ds, params)

    if args.test:
        max_length = params['max_length']
        cuda_device = params['cuda_device']
        model_type = params['model_type']
        test_file_path = params['test_file_path']
        model_output_path = params['model_output_path']
        result_file_path = params['result_file_path']
        debug = params['debug']

        test_ds = task_func(dataset_path=test_file_path, tokenizer=tokenizer, max_length=max_length, debug=debug,
                            test=True)

        test_sentences = test_ds['sentence'].tolist()

        model.test(test_sentences=test_sentences, params=params)
