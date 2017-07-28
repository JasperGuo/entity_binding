# coding=utf8

import numpy as np
import json
import codecs


"""
ID:
    0 - PAD
    1 - UNK
"""


def read_glove(file, dim=50):
    # Add Unknown
    # Reserve 0 to PAD
    word_id = 2
    word_dict = {
        "UNK": 1
    }
    word_vector_matrix = [[0.0] * dim]
    with codecs.open(file, "r", encoding="utf8") as f:
        for line in f:
            content = line.strip().split(" ")
            word = content[0]
            vector = content[1:]
            assert len(vector) == dim

            if word in word_dict:
                print(word)

            word_dict[word] = word_id
            word_vector_matrix.append(vector)
            word_id += 1
    np.save("word_embedding.npy", np.array(word_vector_matrix))

    with open("word_dict.json", "w") as f:
        f.write(json.dumps(word_dict))
    print(len(word_vector_matrix))


def prepare_character_vocab(file):
    """
    Reserve 0 to PAD
    :param file:
    :return:
    """
    char_id = 1
    char_dict = dict()
    with open(file, "r") as f:
        for line in f:
            if line.strip() in char_dict:
                print(line.strip())
            char_dict[line.strip()] = char_id
            char_id += 1
    with open("char_dict.json", "w") as f:
        f.write(json.dumps(char_dict))


def prepare_data_type_vocab(file):
    data_type_id = 0
    data_type_dict = dict()
    with open(file, "r") as f:
        for line in f:
            data_type_dict[line.strip()] = data_type_id
            data_type_id += 1

    with open("data_type.json", "w") as f:
        f.write(json.dumps(data_type_dict))


if __name__ == "__main__":
    read_glove(".\\vocab\\glove.6B\\glove.6B.50d.txt")
    # prepare_character_vocab(".\\vocab\\character.txt")
    # prepare_data_type_vocab(".\\vocab\\data_type.txt")


