def read_prompt_file(prompt_file):
    with open(prompt_file, 'r') as file:
        prompt_msg = file.read()
        return prompt_msg
