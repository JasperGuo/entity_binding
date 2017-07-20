# coding=utf8
import os
import re
import json
from bson import json_util
from util import read_file
from preprocess_table import process_value

COLUMN_PATTERN = re.compile("^col__(.*?)__\((\d+), (\d+)\)$")
CELL_PATTERN = re.compile("^cell__(.*?)__\((\d+), (\d+)\)$")
TABLE_PATTERN = re.compile("^TABLE__(.*)$")


def get_table_map_id(table_loc):
    return '/'.join(table_loc).replace(".csv", ".table")


def process_labeled_sequence(sequence):
    labeled_sequence = list()
    for s in sequence:
        begin = s[0]
        end = s[1]
        assert begin <= end
        label = s[2].strip()
        i = begin
        while i <= end:
            labeled_sequence.append(label)
            i += 1
    return labeled_sequence


def get_labeled_sequence_index(sequence, table):
    """
    PAT: 0
    LIT: 1
    TAB: 2
    COLUMN_NAME:
    COLUMNS:
    :param sequence:
    :param table:
    :return:
    """
    columns = table["columns"]
    label_index = list()
    for label in sequence:
        if label.lower() == "pat":
            label_index.append(0)
        else:
            match = COLUMN_PATTERN.search(label)
            if match:
                # COL
                value = process_value(match.group(1))
                column_idx = match.group(3)
                actual_idx = 2 + 1 + int(column_idx)
                label_index.append(actual_idx)
            else:
                # CELL
                cell_match = CELL_PATTERN.search(label)
                if cell_match:
                    value = process_value(cell_match.group(1))
                    column_idx = int(cell_match.group(3))
                    offset = 0
                    for i in range(column_idx):
                        offset += len(columns[i])
                    offset += (2 + 1 + len(table["column_name"]))
                    index = columns[column_idx].index(value)
                    actual_idx = offset + index
                    label_index.append(actual_idx)
                else:
                    # Table
                    table_match = TABLE_PATTERN.search(label)
                    if table_match:
                        actual_idx = 2
                        label_index.append(actual_idx)
                    else:
                        actual_idx = 1
                        label_index.append(actual_idx)
    return label_index


def process_question(question, table_dict):
    tokenized_sentence = question["tokenized_sentence"].strip().split("|")
    table_map_id = get_table_map_id(question["table_loc"])
    labeled_sequence = process_labeled_sequence(question["bound_sequence"])
    question_id = question["_id"]
    table = table_dict[table_map_id]
    label_sequence_index = get_labeled_sequence_index(labeled_sequence, table)

    return {
        "tokenized_sentence": tokenized_sentence,
        "table_map_id": table_map_id,
        "labeled_sequence": labeled_sequence,
        "question_id": question_id,
        "label_sequence_index": label_sequence_index
    }


def main(file, table_file):
    questions = read_file(file)
    tables = read_file(table_file)

    table_dict = dict()
    for t in tables:
        table_dict[t["map_id"]] = t

    result = list()
    for q in questions:
        try:
            result.append(json.dumps(process_question(q, table_dict), default=json_util))
        except Exception as e:
            print(e)
            continue

    file_base_name = os.path.basename(file)
    dirname = os.path.dirname(file)
    with open(os.path.join(dirname, "preprocessed_" + file_base_name), "w") as f:
        f.write('\n'.join(result))
    print(len(result))


if __name__ == "__main__":
    main("training_questions.txt", "preprocessed_training_tables.txt")
