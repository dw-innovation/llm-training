OUTPUT_FOLDER=results/spot/

#echo zero shot prompt with simple prompt
#
#mkdir -p $OUTPUT_FOLDER
#
#python -m app.nshot.chatgpt_zero_shot \
#--prompt_file scripts/spot/prompt/zero_shot_prompt.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_chatgpt_zeroshot.jsonl
##
#echo zero shot prompt with simple cot prompt
#
#python -m app.nshot.chatgpt_zero_shot \
#--prompt_file scripts/spot/prompt/zero_shot_cot_prompt.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_chatgpt_cot_zeroshot.jsonl


#echo zero shot prompt human prompt v1
#
#for i in {1..10}
#do
#python -m app.nshot.chatgpt_zero_shot \
#--prompt_file scripts/spot/prompt/zero_shot_human_prompt_v1.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_chatgpt_zeroshot_human_prompt_v1_file_${i}.jsonl
#done

echo zero shot prompt human prompt v2

for i in {1..10}
do
model_name=zero_shot_human_prompt_v2
python -m app.nshot.chatgpt_zero_shot \
--prompt_file scripts/spot/prompt/${model_name}.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_${model_name}_file_${i}.jsonl
done