import pandas as pd
import evaluate
from argparse import ArgumentParser


def display_rouge(result_file):
    rouge = evaluate.load('rouge')
    results = pd.read_csv(result_file, sep='\t')
    print(rouge.compute(predictions=results["generated_sentence"],
                        references=results["expected_sentence"]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--results_file")

    args = parser.parse_args()
    display_rouge(args.results_file)
