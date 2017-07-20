# coding=utf8


"""
Max Match algorithm to look up table
Forward & Backward

"Computation of Normalized Edit Distance and Applications"

"""

from nltk import word_tokenize
from util import read_file


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


def build_unmerged_vocab_list(lemma_type, table):
    table_name_key = ''.join([lemma_type, "_table_name"])
    column_name_key = ''.join([lemma_type, "_column_name"])
    columns_key = ''.join([lemma_type, "_columns"])

    vocab_list = [table[table_name_key]]
    for column_name in table[column_name_key]:
        vocab_list.append(column_name)
    for columns in table[columns_key]:
        for cell in columns:
            vocab_list.append(cell)
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

    return len(intersect)/len(union_set), len(str1_set - intersect)/len(union_set)


def prob_forward_match(vocab_list, words):
    """
    Probabilistic forward max match
    :param vocab_list:
    :param words:
    :return:
    """
    result = list()
    i = 0
    # print(vocab_list)
    while i < len(words):
        j = len(words)
        while i <= j <= len(words):
            sub_string = ' '.join(words[i:j])
            sub_string_words = word_tokenize(sub_string)
            max_match = [None, 0.0]
            for idx, vocab in enumerate(vocab_list):
                vocab_words = word_tokenize(vocab)
                score, threshold = jaccard_index(sub_string_words, vocab_words)
                if sub_string in vocab and score > max(threshold, len(sub_string_words)/4*len(vocab_words)):
                    if max_match[1] < score:
                        max_match[0] = idx
                        max_match[1] = score

            if not max_match[0]:
                j -= 1
            else:
                # Match
                result.append((i, j - 1, vocab_list[max_match[0]], True))
                print(sub_string, "M")
                i = j
                break
        else:
            result.append((i, i, False))
            print(words[i])
            i += 1
    return result


def cmp_forward_and_backward(fw_result, bw_result):
    """
    Less match is better,
    :param fw_result:
    :param bw_result:
    :return:
    """
    total_count = len(fw_result) - len(bw_result)
    non_vocab_word = len([m[2] for m in fw_result if not m[2]]) - len([m[2] for m in bw_result if not m[2]])
    single_word = len([1 for m in fw_result if m[0] == m[1] and m[2]]) - len([1 for m in bw_result if m[0] == m[1] and m[2]])

    if total_count + non_vocab_word + single_word > 0:
        return bw_result
    else:
        return fw_result


def construct_match_matrix(match, sentence, vocab_list):
    """
    :param match:
    :param table:
    :param vocab_list:
    :return:   [len(S), Size(table)]
    """
    matrix = [[] for i in range(len(sentence))]
    for m in match:
        if not m[2]:
            continue
        sub_string = ' '.join(sentence[m[0]:m[1]+1])
        idx_list = list()
        for idx, word in enumerate(vocab_list):
            if sub_string == word:
                idx_list.append(idx)
        for idx in idx_list:
            i = m[0]
            while i <= m[1]:
                matrix[i].append(idx)
                i += 1
    return matrix


def match(question, table_dict):
    table = table_dict[question["table_map_id"]]

    vocab_list = build_vocab_list("lemma", table)
    # unmerged_vocab_list = build_unmerged_vocab_list("lemma", table)

    fw_result = forward_match(vocab_list, question["lemma"])
    prob_fw_result = prob_forward_match(vocab_list, question["lemma"])
    # bw_result = backward_match(vocab_list, question["lemma"])
    #
    # match = cmp_forward_and_backward(fw_result, bw_result)
    # sentence = question["lemma"]
    # matrix = construct_match_matrix(match, sentence, vocab_list)
    # print(sentence)
    # print(match)
    # print(matrix)
    # print(question["question_id"])
    print(fw_result)
    print(prob_fw_result)
    print(question["question_id"])
    print("==================================================")


def main(question_file, table_file):
    questions = read_file(question_file)
    tables = read_file(table_file)

    table_dict = dict()
    for table in tables:
        table_dict[table["map_id"]] = table

    for question in questions:
        match(question, table_dict)
    # match(questions[500], table_dict)


if __name__ == "__main__":
    main("lemmatized_preprocessed_training_questions.txt", "lemmatized_preprocessed_training_tables.txt")