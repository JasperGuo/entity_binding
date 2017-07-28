# coding=utf8

"""
Based on model log, extract value from table
"""
import json
import codecs
import os
import re
import argparse

LOG_PATTERN = re.compile("id:\s(.*)\ntid:\s(.*)\nmax_column:\s(\d+)\nmax_cell_value_per_col:\s(\d+)\nt:\s(.*)\np:\s(.*)\nResult:\s(False|True)\n")


def read_log(log_file):
    with open(log_file, "r") as f:
        content = f.read()
        questions = LOG_PATTERN.finditer(content)

    result = list()
    for q in questions:
        result.append({
            "qid": q.group(1),
            "tid": q.group(2),
            "max_column_num": int(q.group(3)),
            "max_cell_value_per_col": int(q.group(4)),
            "g": [int(i) for i in q.group(5).replace(" ", "").split(",")],
            "p": [int(i) for i in q.group(6).replace(" ", "").split(",")],
            "r": q.group(7)
        })
    return result


def read_tables(table_file):
    content = list()
    with open(table_file) as f:
        for line in f:
            content.append(json.loads(line.strip()))
    return content


def build_table_dict(tables):
    table_dict = dict()
    for t in tables:
        table_dict[t["map_id"]] = t
    return table_dict


def process(question, table):
    max_column_num = question["max_column_num"]
    max_cell_value_per_col = question["max_cell_value_per_col"]
    ground_truth_index = question["g"]
    prediction_index = question["p"]
    flatten_table = ["PAT", "LIT", table["table_name"]] + table["column_name"] + ["-"] * (max_column_num - len(table["column_name"]))
    for col in table["columns"]:
        flatten_table += col
        flatten_table += ["-"] * (max_cell_value_per_col - len(col))
    ground_truth_value = list()
    prediction_value = list()
    for g_idx, p_idx in zip(ground_truth_index, prediction_index):
        ground_truth_value.append((g_idx, flatten_table[g_idx],))
        prediction_value.append((p_idx, flatten_table[p_idx],))
    question.update({
        "g_value": ground_truth_value,
        "p_value": prediction_value,
        "table": flatten_table
    })


def save(questions, file):
    with codecs.open(file, "w", encoding="utf8") as f:
        string = ""
        for q in questions:
            string += "=======================\n"
            for key in ["qid", "tid", "max_column_num", "max_cell_value_per_col", "r", "g", "p", "g_value", "p_value", "table"]:
                value = q[key]
                if isinstance(value, list):
                    temp = ','.join([str(i) for i in value])
                else:
                    temp = str(value)
                string += (key + ": " + temp + "\n")
        f.write(string)


def main(log_file, table_file):
    """
    :param log_file:
    :param table_file:
    :return:
    """
    tables = read_tables(table_file)
    table_dict = build_table_dict(tables)
    questions = read_log(log_file)
    for q in questions:
        process(q, table_dict[q["tid"]])
    file_base_name = os.path.basename(log_file)
    dirname = os.path.dirname(log_file)
    file = os.path.join(dirname, "processed_" + file_base_name)
    save(questions, file)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log", help="log file", required=True)
    arg_parser.add_argument("--table", help="table file", required=True)
    args = arg_parser.parse_args()
    main(args.log, args.table)
