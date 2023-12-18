OUTPUT_FOLDER=results/spot/

echo zero shot prompt with simple prompt

mkdir -p $OUTPUT_FOLDER

python -m app.nshot.chatgpt_zero_shot \
--prompt_file scripts/spot/prompt/zero_shot_prompt.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_chatgpt_zeroshot.jsonl

echo zero shot prompt version_2

python -m app.nshot.chatgpt_zero_shot \
--prompt_file scripts/spot/prompt/zero_shot_prompt_v2.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_chatgpt_zeroshot_v2.jsonl

echo zero shot prompt with simple cot prompt

python -m app.nshot.chatgpt_zero_shot \
--prompt_file scripts/spot/prompt/zero_shot_cot_prompt.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_chatgpt_cot_zeroshot.jsonl

echo zero shot cot prompt version_2

python -m app.nshot.chatgpt_zero_shot \
--prompt_file scripts/spot/prompt/zero_shot_cot_prompt_v2.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_chatgpt_cot_zeroshot_v2.jsonl