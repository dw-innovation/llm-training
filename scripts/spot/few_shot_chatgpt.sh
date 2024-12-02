OUTPUT_FOLDER=results/spot/

echo zero shot prompt with simple prompt

mkdir -p $OUTPUT_FOLDER

#python -m app.nshot.chatgpt_zero_shot \
#--prompt_file scripts/spot/prompt/zero_shot_prompt.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_chatgpt_zeroshot.jsonl
#
echo zero shot prompt with simple cot prompt

MODEL_NAME=gpt-4o

python -m app.nshot.chatgpt_zero_shot \
--prompt_file scripts/spot/prompt/zero_shot_cot_prompt.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_${MODEL_NAME}_cot_fewshot_1_2.jsonl \
--few_shot 1
