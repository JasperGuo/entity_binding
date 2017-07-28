# coding=utf8


"""
Max Match algorithm to look up table
Forward & Backward

"Computation of Normalized Edit Distance and Applications"

"""

import json
import os
import random
from nltk import word_tokenize
from util import read_file
from preprocess_table import check_value_type


def build_vocab_list(lemma_type, table):
    table_name_key = ''.join([lemma_type, "_table_name"])
    column_name_key = ''.join([lemma_type, "_column_name"])
    columns_key = ''.join([lemma_type, "_columns"])

    vocab_list = [' '.join(table[table_name_key])]
    for column_name in table[column_name_key]:
        vocab_list.append(' '.join(column_name))
    for columns in table[columns_key]:
        for cell in columns:
            vocab_list.append(' '.join(cell))
    return vocab_list


def forward_match(vocab_list, words):
    """
    Forward max match
    :param vocab_list:
    :param words:
    :return:
    """
    result = list()
    i = 0
    while i < len(words):
        j = len(words)
        while i <= j <= len(words):
            sub_string = ' '.join(words[i:j])
            if sub_string not in vocab_list:
                j -= 1
            else:
                # Match
                result.append((i, j-1, True))
                # print(sub_string, "M")
                i = j
                break
        else:
            result.append((i, i, False))
            # print(words[i])
            i += 1
    return result


def backward_match(vocab_list, words):
    """
    Backward max match
    :param vocab_list:
    :param words:
    :return:
    """
    result = list()
    i = len(words)
    while i >= 0:
        j = 0
        while j < i:
            sub_string = ' '.join(words[j:i])
            if sub_string not in vocab_list:
                j += 1
            else:
                # Match
                result.append((j, i-1, True))
                # print(sub_string, "M")
                i = j
                break
        else:
            i -= 1
            if i >= 0:
                result.append((i, i, False))
                # print(words[i])
    return result


def jaccard_index(str1, str2):
    str1_set = set(str1)
    str2_set = set(str2)

    intersect = str2_set & str1_set
    union_set = str1_set | str2_set

    if len(union_set) == 0:
        print(str1, str1_set)
        print(str2, str2_set)
        return 0, 1, 1

    return len(intersect)/len(union_set), len(str1_set - intersect)/len(union_set), len(str2_set - intersect)/len(union_set)


def prob_forward_match(vocab_list, words):
    """
    Probabilistic forward max match
    :param vocab_list:
    :param words:
    :return:
    """
    result = list()
    i = 0
    while i < len(words):
        j = len(words)
        while i <= j <= len(words):
            sub_string = ' '.join(words[i:j])
            sub_string_words = word_tokenize(sub_string)
            max_match = [None, 0.0]
            same_score_list = list()
            for idx, vocab in enumerate(vocab_list):
                vocab_words = word_tokenize(vocab)
                score, threshold1, threshold2 = jaccard_index(sub_string_words, vocab_words)
                if sub_string in vocab and score > max(threshold1, threshold2):
                    if max_match[1] < score:
                        max_match[0] = idx
                        max_match[1] = score
                        same_score_list = [idx]
                    elif max_match[1] == score:
                        same_score_list.append(idx)

            if max_match[0] is None:
                j -= 1
            else:
                # Match
                result.append((i, j-1, True, max_match[1], same_score_list))
                print(sub_string, "M", vocab_list[max_match[0]])
                i = j
                break
        else:
            result.append((i, i, False))
            # print(words[i])
            i += 1
    return result


def prob_backward_match(vocab_list, words):
    """
    Probabilistic forward max match
    :param vocab_list:
    :param words:
    :return:
    """
    result = list()
    i = len(words)
    while i >= 0:
        j = 0
        while j < i:
            sub_string = ' '.join(words[j:i])
            sub_string_words = word_tokenize(sub_string)
            if len(sub_string_words) == 0:
                continue
            max_match = [None, 0.0]
            same_score_list = list()
            for idx, vocab in enumerate(vocab_list):
                vocab_words = word_tokenize(vocab)
                score, threshold1, threshold2 = jaccard_index(sub_string_words, vocab_words)
                if sub_string in vocab and score >= max(threshold1, threshold2):
                    if max_match[1] < score:
                        max_match[0] = idx
                        max_match[1] = score
                        same_score_list = [idx]
                    elif max_match[1] == score:
                        same_score_list.append(idx)

            if not max_match[0]:
                j += 1
            else:
                # Match
                result.append((j, i-1, True, max_match[1], same_score_list))
                print(sub_string, "M", vocab_list[max_match[0]])
                i = j
                break
        else:
            i -= 1
            if i >= 0:
                result.append((i, i, False))
                # print(words[i])
    return result


def cmp_forward_and_backward(fw_result, bw_result):
    """
    :param fw_result:
    :param bw_result:
    :return:
    """

    fw_match_length = 0
    for m in fw_result:
        if not m[2]:
            continue
        l = m[1] + 1 - m[0]
        fw_match_length += l

    bw_match_length = 0
    for m in bw_result:
        if not m[2]:
            continue
        l = m[1] + 1 - m[0]
        bw_match_length += l

    if fw_match_length > bw_match_length:
        # print("fw")
        return fw_result
    else:
        # print("bw")
        return bw_result


def construct_match_matrix(match, sentence):
    """
    :param sentence:
    :param match:
    :return:   [len(S), Size(table)]
    """
    matrix = [[] for i in range(len(sentence))]
    for m in match:
        if not m[2]:
            continue
        for idx in m[-1]:
            i = m[0]
            while i <= m[1]:
                # idx + 2: insert special tag at the beginning, PAT, LIT
                matrix[i].append((idx+2, m[3]))
                i += 1
    return matrix


def expand_matrix(matrix, sentence):
    for idx, word in enumerate(sentence):
        value_type = check_value_type(word)
        if value_type == "DATE" or value_type == "NUMERIC":
            # 1 -- stands for LIT
            matrix[idx].insert(0, (1, 1.0))
    return matrix


def match_table(question, table_dict):
    table = table_dict[question["table_map_id"]]

    vocab_list = build_vocab_list("stem", table)

    # fw_result = forward_match(vocab_list, question["stem"])
    # bw_result = backward_match(vocab_list, question["stem"])
    prob_fw_result = prob_forward_match(vocab_list, question["stem"])
    print("=================")
    prob_bw_result = prob_backward_match(vocab_list, question["stem"])
    result = cmp_forward_and_backward(prob_fw_result, prob_bw_result)
    sentence = question["stem"]
    matrix = construct_match_matrix(result, sentence)
    matrix = expand_matrix(matrix, question["stem"])
    question.update({
        "table_lookup": matrix
    })
    print(prob_bw_result)
    print(prob_fw_result)
    print(question["question_id"])
    print(matrix)
    # print("==================================================")
    return question


def main(question_file, table_file, _id=None):
    questions = read_file(question_file)
    tables = read_file(table_file)

    table_dict = dict()
    for table in tables:
        table_dict[table["map_id"]] = table

    if _id:
        for q in questions:
            if q["question_id"]["$oid"] == _id:
                match_table(q, table_dict)
    else:
        result = list()
        for question in questions:
            result.append(json.dumps(match_table(question, table_dict)))
        print(len(result))
        file_base_name = os.path.basename(question_file)
        dirname = os.path.dirname(question_file)
        save_file = os.path.join(dirname, "table_lookup_" + file_base_name)
        with open(save_file, "w") as f:
            f.write('\n'.join(result))


if __name__ == "__main__":
    main("lemmatized_preprocessed_training_questions.txt", "lemmatized_preprocessed_training_tables.txt")