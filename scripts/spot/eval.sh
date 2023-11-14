#echo model v9
#python -m app.eval \
#--results_file results/t5_tuned_base_minimized_v3_db-v9_output.tsv
#
#echo model v10
#python -m app.eval \
#--results_file results/t5_tuned_base_minimized_v3_db-v10_output.tsv
#
#
echo model v2
python -m app.eval \
--results_file results/t5_tuned_base_minimized_v2_db-v10_output.tsv

echo model v1 mt5
python -m app.eval \
--results_file results/mt5_tuned_base_minimized_v1_db-v10_output.tsv