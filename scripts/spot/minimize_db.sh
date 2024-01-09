#echo Construct the datasets for analysis/training the models with default output
#
#DATASET_VERSION=v10
#DATASET_FOLDER=tasks/spot/${DATASET_VERSION}
#
#python -m scripts.spot.adopt_db \
#--input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_dev_ChatNL.csv \
#--output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_dev_ChatNL_minimized.csv
#
#python -m scripts.spot.adopt_db \
#--input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_test_ChatNL.csv \
#--output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_test_ChatNL_minimized.csv
#
#python -m scripts.spot.adopt_db \
#--input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_train_ChatNL.csv \
#--output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_train_ChatNL_minimized.csv

#echo merge datasets
#python -m app.analysis.merge \
#--input_file dataset/IMR_Dataset_v4_ChatNL_minimized.csv,dataset/IMR_Dataset_v5_ChatNL_minimized.csv \
#--output_file dataset/IMR_Dataset_v4v5_ChatNL_minimized.csv


# echo Construct the datasets for analysis/training the models with yaml output

# DATASET_VERSION=v10
# DATASET_FOLDER=tasks/spot/${DATASET_VERSION}
# OUTPUT_TYPE=yaml_out

# python -m scripts.spot.adopt_db \
# --input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_dev_ChatNL.csv \
# --output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_dev_ChatNL_minimized_${OUTPUT_TYPE}.csv \
# --output_type yaml

# python -m scripts.spot.adopt_db \
# --input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_test_ChatNL.csv \
# --output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_test_ChatNL_minimized_${OUTPUT_TYPE}.csv \
# --output_type yaml

# python -m scripts.spot.adopt_db \
# --input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_train_ChatNL.csv \
# --output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_train_ChatNL_minimized_${OUTPUT_TYPE}.csv \
# --output_type yaml

echo Construct the datasets for analysis/training the models with yaml output

DATASET_VERSION=v10.1
DATASET_FOLDER=tasks/spot/${DATASET_VERSION}
OUTPUT_TYPE=yaml_out

python -m scripts.spot.adopt_db \
--input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_dev_ChatNL.csv \
--output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_dev_ChatNL_minimized_${OUTPUT_TYPE}.csv \
--output_type yaml

python -m scripts.spot.adopt_db \
--input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_test_ChatNL.csv \
--output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_test_ChatNL_minimized_${OUTPUT_TYPE}.csv \
--output_type yaml

python -m scripts.spot.adopt_db \
--input_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_train_ChatNL.csv \
--output_file ${DATASET_FOLDER}/IMR_Dataset_${DATASET_VERSION}_train_ChatNL_minimized_${OUTPUT_TYPE}.csv \
--output_type yaml