echo model v9
python -m app.eval \
--results_file results/t5_tuned_base_minimized_v3_db-v9_output.tsv

echo model v10
python -m app.eval \
--results_file results/t5_tuned_base_minimized_v3_db-v10_output.tsv