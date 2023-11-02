# LLM Training
This repository is intended to use for training HuggingFace models for the ReCo projects. 


## Installation with Docker
Build docker and strrt training as follows:
```shell
docker build -t llm_training:latest .
screen -L -Logfile t5_train sudo docker run --rm --gpus all -v /reco/llm_training/:/app --name llm_training llm_training:latest bash scripts/spot/train_t5.sh
```

run the following commmand, because we have a limited space and don't want to occupy unncessary spaces

```shell
sudo docker rmi $(docker images -f "dangling=true" -q) --force
```