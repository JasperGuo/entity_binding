# coding=utf8

import json


def read_file(file):
    content = list()
    with open(file) as f:
        for line in f:
            content.append(json.loads(line.strip()))
    return content
