# coding=utf8

import os
import re
import json
import argparse
from bson import json_util

NUMBER_PATTERN1 = re.compile("^\d*\.?\d*$")
NUMBER_PATTERN2 = re.compile("^([0-9,]+)$")
YEAR_PATTERN = re.compile("^[12][0-9]{3}$")
MONTH_PATTERN = re.compile("(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|may|june|july|august|september|october|november|december)")


def read_file(file):
    tables = list()
    with open(file) as f:
        for line in f:
            tables.append(json.loads(line.strip()))
    return tables


def process_value(value):
    value = value.strip().lower()
    if len(value) == 0:
        value = "-"
    if "%" in value:
        value = value.replace("%", " percentage ")
    if "#" in value:
        value = value.replace("#", " number ")
    # Replace multiple spaces with single space
    value = value.replace("'", "")
    value = value.replace('"', "")
    value = value.replace("(", "")
    value = value.replace(")", "")
    value = re.sub(' +', ' ', value)
    return value


def group_table_by_column(rows):
    assert len(rows) > 0
    column_num = len(rows[0])
    columns = [[] for i in range(column_num)]
    for r in rows:
        for i in range(column_num):
            value = process_value(r[i])
            columns[i].append(value)
    return columns


def check_value_type(value):
    if YEAR_PATTERN.search(value):
        return "DATE"
    elif MONTH_PATTERN.search(value):
        return "DATE"
    elif NUMBER_PATTERN1.search(value):
        return "NUMERIC"
    elif NUMBER_PATTERN2.search(value):
        return "NUMERIC"
    return "STRING"


def check_column_type(column_value_type):
    data_type = dict()
    for t in column_value_type:
        if t not in data_type:
            data_type[t] = 0
        data_type[t] += 1

    max_type = ("", -1)
    for t, count in data_type.items():
        if count > max_type[1]:
            max_type = (t, count)

    return max_type[0]


def process_table(table):
    columns = group_table_by_column(table["rows"])

    clean_columns = list()
    column_name = list()
    for c in columns:
        column_name.append(c.pop(0))
        clean_columns.append(list(set(c)))

    column_type = list()
    for c in clean_columns:
        data_type = list()
        for value in c:
            data_type.append(check_value_type(value))
        column_type.append(check_column_type(data_type))
    """
    print(table["_id"])
    print("column_name: ", column_name)
    print("column_type: ", column_type)
    print("columns: ", clean_columns)
    """

    table_info = {
        "columns": clean_columns,
        "column_name": column_name,
        "column_type": column_type,
        "table_name": process_value(table["title"]),
        "map_id": table["map_id"],
        "_id": table["_id"]
    }

    return json.dumps(table_info, default=json_util.default)


def main(file):
    tables = read_file(file)
    result = list()
    for table in tables:
        try:
            result.append(process_table(table))
        except Exception as e:
            print(e)
    file_base_name = os.path.basename(file)
    dirname = os.path.dirname(file)
    with open(os.path.join(dirname, "preprocessed_" + file_base_name), "w") as f:
        f.write('\n'.join(result))
    print(len(result))

if __name__ == "__main__":
    # arg_parser = argparse.ArgumentParser("Path of tables file")
    # arg_parser.add_argument("--file", help="path")
    # args = arg_parser.parse_args()
    main("test_tables.txt")
