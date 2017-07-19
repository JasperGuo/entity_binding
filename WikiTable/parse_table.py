# coding=utf8

"""
Store table in mongodb
"""

import json
import codecs
import re
import os
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client.wiki_table_questions

TABLE_FILE_PATTERN = re.compile(".table")
JSON_FILE_PATTERN = re.compile(".json")

DATA_PATH = os.path.join(os.curdir, "data")
PAGE_PATH = os.path.join(os.curdir, "page")
DIR_LIST = ["200", "201", "202", "203", "204"]
# DIR_LIST = ["200"]


def mongo_save(result):
    db.tables.insert_many(result)
    print(db.tables.count())


def main():
    table_info = dict()
    for directory in DIR_LIST:
        curr_data_path = os.path.join(DATA_PATH, '-'.join([directory, "csv"]))
        curr_page_path = os.path.join(PAGE_PATH, '-'.join([directory, "page"]))

        for file in os.listdir(curr_data_path):

            if not TABLE_FILE_PATTERN.search(file):
                continue
            filename, ext = os.path.splitext(file)
            json_file = os.path.join(curr_page_path, ''.join([filename, ".json"]))
            if not os.path.exists(json_file):
                continue
            with open(json_file, "r") as f:
                table_meta = json.load(f)
            rows = list()
            table_file = os.path.join(curr_data_path, file)
            with codecs.open(table_file, "r", "utf8") as f:
                for line in f:
                    line = line.strip()
                    line = line.split("|")[1:-1]
                    _ = [word.strip() for word in line]
                    rows.append(_)
            index = ["csv", '-'.join([directory, "csv"]), file]
            info = {
                "rows": rows,
                "path": index
            }
            info.update(table_meta)
            table_info.update({
                '/'.join(index): info
            })
    with open("table_info.json", "w") as f:
        f.write(json.dumps(table_info, sort_keys=True, indent=4))
    result = list()
    for key, value in table_info.items():
        value.update({
            "map_id": key
        })
        result.append(value)
    mongo_save(result)


if __name__ == "__main__":
    main()
