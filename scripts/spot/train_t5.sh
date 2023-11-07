MODEL_TYPE=t5
TASK=spot
PRETRAINED_MODEL=t5-base
MODEL_OUT=model/t5_tuned_base_minimized_v3_db-v9
MAX_LENGTH=1024
EVAL_METRIC=eval_rouge2
RESULT_FILE_PATH=results/t5_tuned_base_minimized_v3_db-v9_output.tsv

LEARNING_RATE=1e-3
EPOCHS=10
RANDOM_SEED=0

TRAIN_DATASET=tasks/spot/v9/IMR_Dataset_v9_train_ChatNL_minimized.csv
VAL_DATASET=tasks/spot/v9/IMR_Dataset_v9_dev_ChatNL_minimized.csv
TEST_DATASET=tasks/spot/v9/IMR_Dataset_v9_test_ChatNL_minimized.csv

CUDA_DEVICE=0

mkdir -p results

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
--test

