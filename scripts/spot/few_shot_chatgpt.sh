OUTPUT_FOLDER=results/spot

#echo 1-shot prompt human prompt v2
#
#for i in {1..10}
#do
#few_shot=1
#model_name=few_shot_prompt_human_prompt_v2
#python -m app.nshot.chatgpt_few_shot \
#--few_shot $few_shot \
#--prompt_file scripts/spot/prompt/${model_name}.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_${model_name}_shot_${few_shot}_file_${i}.jsonl
#done

#echo 2-shot prompt human prompt v2
#
#for i in {1..10}
#do
#few_shot=2
#model_name=few_shot_prompt_human_prompt_v2
#python -m app.nshot.chatgpt_few_shot \
#--few_shot $few_shot \
#--prompt_file scripts/spot/prompt/${model_name}.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_${model_name}_shot_${few_shot}_file_${i}.jsonl
#done
#
#echo 5-shot prompt human prompt v2
#
#for i in {1..10}
#do
#few_shot=5
#model_name=few_shot_prompt_human_prompt_v2
#python -m app.nshot.chatgpt_few_shot \
#--few_shot $few_shot \
#--prompt_file scripts/spot/prompt/${model_name}.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_${model_name}_shot_${few_shot}_file_${i}.jsonl
#done

#echo 10-shot prompt human prompt v2
#
#for i in {2..10}
#do
#few_shot=10
#model_name=few_shot_prompt_human_prompt_v2
#python -m app.nshot.chatgpt_few_shot \
#--few_shot $few_shot \
#--prompt_file scripts/spot/prompt/${model_name}.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_${model_name}_shot_${few_shot}_file_${i}.jsonl
#done