MODEL_TYPE=llama2
TASK=spot
PRETRAINED_MODEL=meta-llama/Llama-2-7b-hf
DATASET_VERSION=v10
MODEL_VERSION=v1
MODEL_OUT=model/${MODEL_TYPE}_tuned_base_minimized_${MODEL_VERSION}_db-${DATASET_VERSION}_yaml_out
MAX_LENGTH=1024
EVAL_METRIC=eval_rouge2
RESULT_FILE_PATH=results/${MODEL_TYPE}_tuned_base_minimized_${MODEL_VERSION}_db-${DATASET_VERSION}_output_yaml_out.jsonl

LEARNING_RATE=2e-5
EPOCHS=2
RANDOM_SEED=0

TRAIN_DATASET=tasks/spot/${DATASET_VERSION}/IMR_Dataset_${DATASET_VERSION}_train_ChatNL_minimized_yaml_out.csv
VAL_DATASET=tasks/spot/${DATASET_VERSION}/IMR_Dataset_${DATASET_VERSION}_dev_ChatNL_minimized_yaml_out.csv

TEST_DATASET="tasks/spot/gold/sentences.txt"

CUDA_DEVICE=0

mkdir -p results
mkdir -p model

echo $MODEL training

python3 -m app.main \
--cuda_device $CUDA_DEVICE \
--train_file_path $TRAIN_DATASET \
--val_file_path $VAL_DATASET \
--test_file_path $TEST_DATASET \
--pretrained_model $PRETRAINED_MODEL \
--model_type $MODEL_TYPE \
--task $TASK \
--learning_rate $LEARNING_RATE \
--epochs $EPOCHS \
--model_output_path $MODEL_OUT \
--random_seed $RANDOM_SEED \
--max_length $MAX_LENGTH \
--eval_metric $EVAL_METRIC \
--result_file_path $RESULT_FILE_PATH \
--test \
--train \
--debug \
--sample_ratio 0.1
