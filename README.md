# LLM Training
This repository is intended to use for training HuggingFace models for the ReCo projects. 


## Installation with Docker
docker build -t llm_training:latest .
sudo docker run --rm --gpus 0 -v ~/Dokumente/Codes/llm-training:/app --name llm_training llm_training:latest bash scripts/train.sh

run the following commmand, because we have a limited space and don't want to occupy unncessary spaces
sudo docker rmi $(docker images -f "dangling=true" -q) --force

scp -r barisschlichti@lovelace:/reco/overpass-ml ./overpass-ml