import pandas as pd
import ast
import json
import re
from argparse import ArgumentParser

RE_PROP_SEP = r'[~=:<>,]+'


def process_output(sample):
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
                    flt["v"] = flt["v"].replace("mm","m")
                if ":" in flt["n"]:
                    flt["n"] = flt["n"].split(":")[-1]
                # flts.append({"n": flt["n"], "op": flt["op"], "v": flt["v"], "k": flt["k"]})
                flts.append({"n": flt["n"], "op": flt["op"], "v": flt["v"], "k": flt["k"]})

        if len(flts) > 0:
            minimized_node["flts"] = flts
        nodes.append(minimized_node)

    minimized_data["ns"] = nodes

    minimized_data = json.dumps(minimized_data)

    minimized_data = minimized_data.replace('{"', '{ "').replace('[{', '[ {').replace('"}', '" }').replace('}]', '} ]').replace(']}', '} ] }').replace("},", "} ,").replace("],", "] ,")

    return minimized_data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--output_file')
    args = parser.parse_args()

    df = pd.read_csv(args.input_file, sep=',')

    df["query"] = df["query"].apply(lambda x: process_output(x))

    # print(df["query"].tolist()[0])

    df["sentence"] = df["sentence"].apply(lambda x: x.replace("\"", "").replace("\n", " "))

    print(f"Number of samples {len(df)}")

    df = df[~df['sentence'].str.contains('''I'm sorry''', flags=re.IGNORECASE, regex=True)]

    print(f"Number of samples after process {len(df)}")

    df = df[["sentence", "query"]]
    df.to_csv(args.output_file, sep="\t", index=False)
