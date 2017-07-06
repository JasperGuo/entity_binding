# coding=utf8

"""
Format database file to multiple csv files.
"""

import re
import os
import argparse
from pprint import pprint

SCHEMA_PATTERN = re.compile("([a-zA-Z]+)\((.*)\)")
ITEM_PATTERN = re.compile("([a-zA-Z]+)\((.*)\).")
COMPLICATE_CELL_VALUE_PATTERN = re.compile("(\[.*?\])")
SCHEMA_END_PATTERN = re.compile("(\*)+/")


def parse_cell_values(values):
    """
    Parse cell values
    :param values:
    :return:
    """
    cell_values = list()
    splits = re.split(COMPLICATE_CELL_VALUE_PATTERN, values)
    for s in splits:
        if '[' in s and ']' in s:
            cell_values.append(':::'.join([_.replace("'", "").strip() for _ in s.replace("[", "").replace("]", "").split(",")]) + ":::")
        else:
            for _ in s.split(","):
                processed = _.replace("'", "").strip()
                if processed:
                    cell_values.append(processed)
    # print(cell_values)
    return cell_values


def save(tables, base_path=""):
    save_path = os.path.join(base_path, "formatted")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for (tab, values) in tables.items():
        with open(os.path.join(save_path, ''.join([tab, ".csv"])), "w") as f:
            string = '\n'.join([','.join(str(_) for _ in v) for v in values])
            f.write(string)


def get_distinct_column_values(tables):
    distinct_values_per_column = dict()
    result = dict()
    for table, values in tables.items():
        distinct_values_per_column[table] = list()
        columns = list()
        for column_name in values[0]:
            columns.append(column_name.strip())
            distinct_values_per_column[table].append(set())
        i = 1
        while i < len(values):
            for idx, v in enumerate(values[i]):
                print(values[i])
                distinct_values_per_column[table][idx].add(v.strip())
            i += 1

        result[table] = dict()
        for idx, c in enumerate(columns):
            result[table][c] = list(distinct_values_per_column[table][idx])
    return result


def main(file):
    tables = dict()
    with open(file, "r") as f:
        is_schema_end = False
        for line in f:

            if not is_schema_end and SCHEMA_END_PATTERN.match(line.strip()):
                is_schema_end = True
            if not is_schema_end:
                scheme_matches = SCHEMA_PATTERN.match(line.strip())
                if scheme_matches:
                    _table = scheme_matches.group(1).strip()
                    _columns = scheme_matches.group(2).strip().split(",")
                    if _table not in tables:
                        tables[_table] = list()
                    tables[_table].append([c.strip() for c in _columns])
                    continue

            matches = ITEM_PATTERN.match(line.strip())
            if matches:
                _table = matches.group(1).strip()
                _cell_values = matches.group(2).strip()
                if _table not in tables:
                    tables[_table] = list()
                _cell_values = parse_cell_values(_cell_values)
                tables[_table].append(_cell_values)
    # pprint(tables)
    base_path = os.path.dirname(os.path.abspath(file))
    # save(tables, base_path)

    pprint(get_distinct_column_values(tables))

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Path of database file")
    arg_parser.add_argument("--file", help="path")
    args = arg_parser.parse_args()
    main(args.file)
