import os
import json
from huggingface_hub import InferenceClient, login
from argparse import ArgumentParser
from transformers import AutoTokenizer
from tqdm import tqdm
from app.nshot.json_extractor import json_extractor
from dotenv import load_dotenv

load_dotenv()
# access token with permission to access the model and PRO subscription
hf_token = os.getenv("HF_INFERENCE_TOKEN")  # https://huggingface.co/settings/tokens
login(token=hf_token)

# tokenizer for generating prompt
tokenizer = AutoTokenizer.from_pretrained(os.getenv("HF_MODEL_TOKENIZER"))

# inference client
client = InferenceClient(os.getenv("HF_MODEL"))


# generate function
def generate(text):
    payload = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
    res = client.text_generation(
        payload,
        do_sample=True,
        return_full_text=False,
        max_new_tokens=2048,
        top_p=0.9,
        temperature=0.6,
    )
    return res.strip()


def read_prompt_file(prompt_file):
    with open(prompt_file, 'r') as file:
        prompt_msg = file.read()
        return prompt_msg


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)

    args = parser.parse_args()

    llama2_prompt = """<s>[INST] <<SYS>>
    {sys_message}
    <</SYS>>
    Text:
    ```{sentence}``` [/INST]
    """

    sys_message = read_prompt_file(prompt_file=args.prompt_file)

    with open(args.input_file, 'r') as input_file:
        sentences = input_file.readlines()

        with open(args.result_file, 'w') as outfile:
            for sentence in tqdm(sentences, total=len(sentences)):
                json_data = {"sentence": sentence}
                sentence = sentence.rstrip()
                generated_text = generate(llama2_prompt.format(sentence=sentence, sys_message=sys_message))
                json_data["model_result"] = generated_text
                json_data["parsed_result"] = json_extractor(generated_text)
                json.dump(json_data, outfile)
                outfile.write('\n')
