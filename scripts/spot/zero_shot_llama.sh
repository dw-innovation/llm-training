OUTPUT_FOLDER=results/spot/

echo zero shot cot prompting

for i in {1..10}
do
model_name=zero_shot_cot_prompt
python -m app.nshot.llama2_zero_shot \
--prompt_file scripts/spot/prompt/${model_name}.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_cot_${model_name}_file_${i}.jsonl
done

echo zero shot prompt with simple prompt

mkdir -p $OUTPUT_FOLDER
#
for i in {1..10}
do
model_name=zero_shot_prompt
python -m app.nshot.llama2_zero_shot \
--prompt_file scripts/spot/prompt/${model_name}.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_zeroshot_${model_name}_file_${i}.jsonl
done


echo zero shot prompt with zero_shot_human_prompt_v2

mkdir -p $OUTPUT_FOLDER
#
for i in {1..10}
do
model_name=zero_shot_human_prompt_v2
python -m app.nshot.llama2_zero_shot \
--prompt_file scripts/spot/prompt/${model_name}.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_zeroshot_${model_name}_file_${i}.jsonl
done