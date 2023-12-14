# LLM Training
This repository is intended to use for training HuggingFace models for the ReCo projects. 


## Installation with Docker
Build docker and start training as follows:

Docker Building:
```shell
docker build -t llm_training:latest .
```

run the following commmand if you can not build its docker, because we have a limited space and don't want to occupy unncessary spaces

```shell
sudo docker rmi $(docker images -f "dangling=true" -q) --force
```

### Fine-tuning

Before training the model, you need to create .env file since we need to use GPU 1. Add `GPU_DEVICE=1` to `.env`.

Model training:
```shell
screen -L -Logfile t5_train sudo docker run --rm --gpus all -v /reco/llm_training/:/app --env-file .env --name llm_training llm_training:latest bash scripts/spot/train_t5.sh
```

Model testing, you need to remove --train from the script:
```shell
screen -L -Logfile t5_train sudo docker run --rm --gpus all -v /reco/llm_training/:/app --env-file .env --name llm_training llm_training:latest bash scripts/spot/train_t5.sh
```


### Zero-shot, few-shot learning

