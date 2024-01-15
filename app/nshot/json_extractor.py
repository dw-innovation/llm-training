import yaml
import logging

# Define a regular expression to find JSON-like structures in the text
ENT_REL_PATTERN = r'area:\s*(.*?)entities:\s*(.*?)relations:\s*(.*?)$'


def json_extractor(input_string):
    if isinstance(input_string, tuple):
        _, input_string = input_string
    json_data = None
    input_string = input_string.replace('```yaml', '').replace('```', '')
    input_string = input_string.replace('*', '-').replace('+', '-')
    input_string = input_string.replace('material:', 'material')

    if "Note" in input_string:
        input_string = input_string.split('Note')[0].strip()

    try:
        json_data = yaml.safe_load(input_string)
    except yaml.parser.ParserError as e:
        logging.error(f"Parser error: {e}")
    except yaml.scanner.ScannerError as e:
        logging.error(f"Scanner error: {e}")
    except yaml.constructor.ConstructorError as e:
        logging.error(f"Construct error: {e}")

    return json_data
