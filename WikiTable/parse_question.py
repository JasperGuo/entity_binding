# coding=utf8

import codecs
import json
import argparse
import re
from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')
db = client.wiki_table_questions

SENTENCE_PATTERN = re.compile("(nt-\d+)\t(.*)\t(csv.*\.csv)\W(.*?)\t(.*?[\?|\.])\t(.*?\?)\W(.*?\.)")


def mongo_save(result):
    db.questions.insert_many(result)
    print(db.questions.count())


def main(file):
    sentences = list()
    fail_sentences = list()
    with codecs.open(file, "r", "utf8") as f:
        for line in f:
            matches = SENTENCE_PATTERN.match(line)
            if matches:
                sentence_id = matches.group(1).strip()
                sentence = matches.group(2).strip()
                table_loc = matches.group(3).strip().split("/")
                answer = matches.group(4).strip()
                tokenized_sentence = matches.group(5).strip()
                pos_tagged_sequence = matches.group(7).strip()
                sentences.append({
                    "is_bound": False,
                    "bound_sequence": "",
                    "bound_date": "",
                    "answer": answer,
                    "sentence_id": sentence_id,
                    "sentence": sentence,
                    "table_loc": table_loc,
                    "tokenized_sentence": tokenized_sentence,
                    "pos_tagged_sequence": pos_tagged_sequence
                })
            else:
                fail_sentences.append(line)
    with open("result.json", "w") as f:
        f.write(json.dumps(sentences, sort_keys=True, indent=4))
    with codecs.open("error.txt", "w", "utf8") as f:
        f.write('\n'.join(fail_sentences))
    print("Success: ", len(sentences))
    print("Fail: ", len(fail_sentences))
    mongo_save(sentences)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser("Path of database file")
    arg_parser.add_argument("--file", help="path")
    args = arg_parser.parse_args()
    main(args.file)
