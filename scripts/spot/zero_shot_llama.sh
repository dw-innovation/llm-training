OUTPUT_FOLDER=results/spot/

echo zero shot prompt with simple prompt

mkdir -p $OUTPUT_FOLDER

python -m app.nshot.llama2_zero_shot \
--prompt_file scripts/spot/prompt/zero_shot_prompt.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_zeroshot.jsonl