import os
import json
from huggingface_hub import InferenceClient, login
from argparse import ArgumentParser
from transformers import AutoTokenizer
from tqdm import tqdm
from app.nshot.json_extractor import json_extractor
from dotenv import load_dotenv
from app.nshot.utils.file_ops import read_prompt_file
from app.nshot.utils.search import search_similar_sentence

load_dotenv()
# access token with permission to access the model and PRO subscription
hf_token = os.getenv("HF_INFERENCE_TOKEN")  # https://huggingface.co/settings/tokens
login(token=hf_token)

# tokenizer for generating prompt
tokenizer = AutoTokenizer.from_pretrained(os.getenv("HF_MODEL_TOKENIZER"))

# inference client
client = InferenceClient(os.getenv("HF_MODEL"))

load_dotenv()


def build_few_shot_prompt(sys_prompt, example_sentences):
    examples_prompt = ''
    for example_sentence in example_sentences:
        sentence = example_sentence['sentence']
        query = example_sentence['query']
        examples_prompt += f'### Input ###\n{sentence}\n### Output ###\n```yaml\n{query}```\n'

    prompt ='<s>[INST] <<SYS>>' + sys_prompt + '<</SYS?>>' +'\n' + examples_prompt

    return prompt


def generate(text):
    payload = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
    res = client.text_generation(
        payload,
        do_sample=True,
        return_full_text=False,
        max_new_tokens=4095,
        top_p=0.9,
        temperature=0.5,
    )
    return res.strip()

if __name__ == '__main__':
    if __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument("--prompt_file", type=str, required=True)
        parser.add_argument("--input_file", type=str, required=True)
        parser.add_argument("--result_file", type=str, required=True)
        parser.add_argument("--few_shot", type=int, required=True)

        args = parser.parse_args()

        # llama2_prompt = """<s>[INST] <<SYS>>
        # {sys_message}
        # <</SYS>>
        # ```{sentence}``` [/INST]
        # """

        sys_message = read_prompt_file(prompt_file=args.prompt_file)

        with open(args.input_file, 'r') as input_file:
            sentences = input_file.readlines()
            sentences = sentences[1:-1]
            with open(args.result_file, 'w') as outfile:
                for sentence in tqdm(sentences, total=len(sentences)):
                    json_data = {"sentence": sentence}
                    sentence = sentence.rstrip()

                    search_results = search_similar_sentence(sentence=sentence, few_shot=args.few_shot)

                    few_shot_prompt = build_few_shot_prompt(sys_prompt=sys_message, example_sentences=search_results)

                    sentence = f"{few_shot_prompt}\n### Input ###\n{sentence}\n### Output ###" + '[/INST]'
                    generated_text = generate(sentence)
                    json_data["model_result"] = generated_text
                    json_data["parsed_result"] = json_extractor(generated_text)
                    json.dump(json_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
