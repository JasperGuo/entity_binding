# coding=utf8

"""
Based on model log, extract value from table
"""
import re
import argparse
from bson.objectid import ObjectId
from pymongo import MongoClient

LOG_PATTERN = re.compile("id:\s(.*)\ntid:\s(.*)\nmax_column:\s(\d+)\nmax_cell_value_per_col:\s(\d+)\nt:\s(.*)\np:\s(.*)\nResult:\s(False|True)\nscores:\s([-|.|\d]+)\n")

client = MongoClient('mongodb://localhost:27017/')
db = client.wiki_table_questions


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
            "r": q.group(7),
            "score": float(q.group(8))
        })
    return result


def update(result):
    bulk = db.questions.initialize_ordered_bulk_op()
    for q in result:
        bulk.find({'_id': ObjectId(q["qid"])}).update({'$set': {'score': q["score"]}})
    result = bulk.execute()
    print(result)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--log", help="log file", required=True)
    args = arg_parser.parse_args()
    result = read_log(args.log)
    update(result)
