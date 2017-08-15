# coding=utf8


"""
Read Labeled training data from MongoDB
"""

import json
from bson import json_util
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client.wiki_table_questions


def get_table_key(table_name_value):
    return '/'.join(table_name_value).replace(".csv", ".table")


def save(file, content):
    with open(file, "w") as f:
        f.write('\n'.join(content))


def query_table(table_key_set):
    tables = db.tables.find(
        {
            "map_id": {
                "$in": list(table_key_set)
            }
        }
    )
    table_str_list = list()
    for table in tables:
        table_str_list.append(json.dumps(table, default=json_util.default))
    save("test_tables.txt", table_str_list)


def fetch_all_table():
    tables = db.tables.find()
    table_str_list = list()
    for table in tables:
        table_str_list.append(json.dumps(table, default=json_util.default))
    save("all_tables.txt", table_str_list)


def main():
    questions = db.test_questions.find({
        "is_bound": True
    })

    table_key_set = set()
    question_str_list = list()
    for question in questions:
        table_key_set.add(get_table_key(question["table_loc"]))
        question_str_list.append(json.dumps(question, default=json_util.default))

    save("test_questions.txt", question_str_list)

    query_table(table_key_set)

    print("Questions: %d" % questions.count())
    print("Tables: %d" % len(table_key_set))


if __name__ == "__main__":
    fetch_all_table()
