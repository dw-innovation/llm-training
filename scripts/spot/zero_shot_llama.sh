OUTPUT_FOLDER=results/spot/

#echo zero shot prompt with simple prompt
#
#mkdir -p $OUTPUT_FOLDER
#
#python -m app.nshot.llama2_zero_shot \
#--prompt_file scripts/spot/prompt/zero_shot_prompt.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_zeroshot.jsonl

#echo zero shot prompt with cot prompt
#
#mkdir -p $OUTPUT_FOLDER
#
#python -m app.nshot.llama2_zero_shot \
#--prompt_file scripts/spot/prompt/zero_shot_cot_prompt.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_cot_zeroshot.jsonl

#python -m app.nshot.llama2_zero_shot \
#--prompt_file scripts/spot/prompt/zero_shot_cot_prompt_v2.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_cot_zeroshot_v2.jsonl

python -m app.nshot.llama2_zero_shot \
--prompt_file scripts/spot/prompt/zero_shot_prompt_v2.txt \
--input_file tasks/spot/gold/sentences.txt \
--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_zeroshot_v2.jsonl



#echo zero shot prompt, triple extraction
#
#mkdir -p $OUTPUT_FOLDER
#
#python -m app.nshot.llama2_zero_shot \
#--prompt_file scripts/spot/prompt/zero_shot_triple_prompt.txt \
#--input_file tasks/spot/gold/sentences.txt \
#--result_file ${OUTPUT_FOLDER}/gold_predictions_llama_triple_zeroshot.jsonl