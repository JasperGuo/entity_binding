# coding=utf8

import json
import numpy as np
import random
import argparse


def read_file(file):
    content = list()
    with open(file) as f:
        for line in f:
            content.append(json.loads(line.strip()))
    return content


def evaluate(question):

    table_matches = question["table_lookup"]
    ground_truth = question["label_sequence_index"]
    prediction = list()
    for match in table_matches:
        if len(match) == 0:
            prediction.append(0)
        else:
            if len(match) == 1:
                prediction.append(match[0][0])
            else:
                if match[0][0] == 1:
                    # LIT
                    prediction.append(random.sample(match[1:], 1)[0][0])
                else:
                    prediction.append(random.sample(match, 1)[0][0])

    assert len(prediction) == len(ground_truth)
    print(question["question_id"])
    print("Pred : ", prediction)
    print("Truth: ", ground_truth)
    print("===========================")

    if np.sum(np.abs(np.array(prediction) - np.array(ground_truth))) == 0:
        return True
    return False


def main(question_file):
    questions = read_file(question_file)
    total = len(questions)
    correct = 0
    for question in questions:
        correct += evaluate(question)

    print("Accuracy: %f" % (correct/total))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Path of question file")
    arg_parser.add_argument("--file", help="path")
    args = arg_parser.parse_args()
    main(args.file)
