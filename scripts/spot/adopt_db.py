import pandas as pd
import ast
import json
import codecs
import re
import yaml
from argparse import ArgumentParser

RE_PROP_SEP = r'[~=:<>,]+'


def decode_unicode(text):
    def unicode_replacer(match):
        return codecs.decode(match.group(0), 'unicode_escape')

    pattern = re.compile(r'\\u[0-9a-fA-F]{4}')
    decoded_text = pattern.sub(unicode_replacer, text)
    decoded_text = decoded_text.replace('\\', '')

    print(decoded_text)

    return decoded_text


def default_process_output(sample):
    data = ast.literal_eval(sample)

    minimized_data = {}
    minimized_data["a"] = data["a"]

    if data["a"]["v"] == "":
        del data["a"]["v"]

    if len(data["es"]) > 0:
        minimized_data["es"] = data["es"]

    if len(data["ns"]) > 0:
        minimized_data["ns"] = []

    nodes = []
    for idx, node in enumerate(data["ns"]):

        minimized_node = {}
        minimized_node["id"] = node["id"]
        # minimized_node["t"] = node["t"]

        minimized_node["n"] = node["flts"][0]["n"]

        flts = []
        if len(node["flts"]) > 1:
            for flt in node["flts"][1:]:
                if flt["k"] == "height" and "mm" in flt["v"]:
                    flt["v"] = flt["v"].replace("mm", "m")
                if ":" in flt["n"]:
                    flt["n"] = flt["n"].split(":")[-1]
                # flts.append({"n": flt["n"], "op": flt["op"], "v": flt["v"], "k": flt["k"]})
                flts.append({"n": flt["n"], "op": flt["op"], "v": flt["v"], "k": flt["k"]})

        if len(flts) > 0:
            minimized_node["flts"] = flts
        nodes.append(minimized_node)

    minimized_data["ns"] = nodes

    minimized_data = json.dumps(minimized_data)

    minimized_data = minimized_data.replace('{"', '{ "').replace('[{', '[ {').replace('"}', '" }').replace('}]',
                                                                                                           '} ]').replace(
        ']}', '} ] }').replace("},", "} ,").replace("],", "] ,")

    return minimized_data


def convert_yaml_output(sample):
    data = ast.literal_eval(sample)

    minimized_data = {}
    minimized_data["area"] = data["a"]

    if isinstance(minimized_data["area"], list):
        print("That is a list!!!!!!")

    else:
        minimized_data["area"]["type"] = minimized_data["area"].pop("t")
        minimized_data["area"]["name"] = minimized_data["area"].pop("v")
        minimized_data["area"]["name"] = decode_unicode(minimized_data["area"]["name"])

    # if data["area"]["value"] == "":
    #     del data["a"]["v"]

    if len(data["es"]) > 0:
        minimized_data["relations"] = data["es"]

        for idx, relation in enumerate(minimized_data["relations"]):
            minimized_data["relations"][idx]["source"] = relation.pop("src")
            minimized_data["relations"][idx]["target"] = relation.pop("tgt")
            minimized_data["relations"][idx]["name"] = relation.pop("t")
            minimized_data["relations"][idx]["value"] = relation.pop("dist")

    if len(data["ns"]) > 0:
        minimized_data["entities"] = []

    nodes = []
    for idx, node in enumerate(data["ns"]):
        minimized_node = {}
        minimized_node["id"] = node["id"]
        # minimized_node["type"] = node["t"]
        minimized_node["name"] = node["flts"][0]["n"]

        flts = []
        if len(node["flts"]) > 1:
            for flt in node["flts"][1:]:
                # if flt["k"] == "height" and "mm" in flt["v"]:
                #     flt["v"] = flt["v"].replace("mm", "m")
                if ":" in flt["n"]:
                    flt["n"] = flt["n"].split(":")[-1]
                flts.append({"name": flt["n"], "operator": flt["op"], "value": flt["v"]})

        if len(flts) > 0:
            minimized_node["filters"] = flts

        nodes.append(minimized_node)

    minimized_data["entities"] = nodes
    minimized_data = yaml.dump(minimized_data, allow_unicode=True)

    return minimized_data


OUTPUT_FUNCS = {
    "default": default_process_output,
    "yaml": convert_yaml_output
}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    parser.add_argument('--output_type', default="default")

    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep=',')

    df["query"] = df["query"].apply(lambda x: OUTPUT_FUNCS[args.output_type](x))

    # print(df["query"].tolist()[0])

    df["sentence"] = df["sentence"].apply(lambda x: x.replace("\"", "").replace("\n", " "))

    print(f"Number of samples {len(df)}")

    df = df[~df['sentence'].str.contains('''I'm sorry''', flags=re.IGNORECASE, regex=True)]

    print(f"Number of samples after process {len(df)}")

    df = df[["sentence", "query"]]

    # df["query"] = df["query"].apply(lambda x: json.dumps(x))

    df.to_csv(args.output_file, sep="\t", index=False)
