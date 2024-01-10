import os
import json
from tqdm import tqdm
from argparse import ArgumentParser
from app.nshot.json_extractor import json_extractor
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from app.nshot.utils import read_prompt_file

load_dotenv()


def generate(client, query, prompt):
    logger.info(f"Prompting {query}")
    completion = client.chat.completions.create(
        model=os.getenv("CHATGPT_MODEL"),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.5,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    logger.info(f"chatgpt returned a result")
    raw_output = completion.choices[0].message.content
    logger.info(f"the document is parsing")
    parsed_output = json_extractor(raw_output)
    logger.info(f"the document parsed")
    return parsed_output, raw_output


if __name__ == '__main__':
    if __name__ == '__main__':
        parser = ArgumentParser()
        parser.add_argument("--prompt_file", type=str, required=True)
        parser.add_argument("--input_file", type=str, required=True)
        parser.add_argument("--result_file", type=str, required=True)

        args = parser.parse_args()

        sys_message = read_prompt_file(prompt_file=args.prompt_file)
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        with open(args.input_file, 'r') as input_file:
            sentences = input_file.readlines()
            sentences = sentences[1:-1]
            with open(args.result_file, 'w') as outfile:
                for sentence in tqdm(sentences, total=len(sentences)):
                    json_data = {"sentence": sentence}
                    sentence = sentence.rstrip()
                    sentence = f"### Input ###\n{sentence}"
                    generated_text = generate(client=client, query=sentence, prompt=sys_message)
                    json_data["model_result"] = generated_text
                    json_data["parsed_result"] = json_extractor(generated_text)
                    json.dump(json_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
