from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM


class T5Model:
    def load(self, pretrained_model):
        return AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)

    def training_args(self, **args):
        return Seq2SeqTrainingArguments(**args)

    def trainer(self, **args):
        return Seq2SeqTrainer(**args)


class LlamaModel:
    pass


MODELS = {
    "t5": T5Model,
    "llama": LlamaModel
}
