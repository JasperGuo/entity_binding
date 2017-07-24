# coding=utf8


from nltk import word_tokenize
import json


"""
ID:
    0 - PAD
    1 - UNK
"""

UNKNOWN_WORD = 1


def read_file(file):
    content = list()
    with open(file) as f:
        for line in f:
            content.append(json.loads(line.strip()))
    return content


def read_vocab(file):
    with open(file, "r") as f:
        return json.load(f)


def lookup_vocab(vocab, word):
    if word in vocab:
        return vocab[word]
    return UNKNOWN_WORD


def table_lookup_vocab(table, word_vocab, char_vocab, data_type_vocab):
    """
    :param table:
    :param word_vocab:
    :param char_vocab:
    :param data_type_vocab:
    :return:
    """
    # table name
    table_name_word_ids = list()
    table_name_char_ids = list()
    table_name_words = word_tokenize(table["table_name"])
    for word in table_name_words:
        table_name_word_ids.append(lookup_vocab(word_vocab, word))
        temp = list()
        for char in word:
            temp.append(lookup_vocab(char_vocab, char))
        table_name_char_ids.append(temp)

    # column name
    column_name_word_ids = list()
    column_name_char_ids = list()
    for cname in table["column_name"]:
        word_ids_temp = list()
        name_char_ids_temp = list()
        cname_words = word_tokenize(cname)
        for word in cname_words:
            word_ids_temp.append(lookup_vocab(word_vocab, word))

            char_ids_temp = list()
            for char in word:
                char_ids_temp.append(lookup_vocab(char_vocab, char))
            name_char_ids_temp.append(char_ids_temp)
        column_name_char_ids.append(name_char_ids_temp)
        column_name_word_ids.append(word_ids_temp)

    # cell value
    column_word_ids = list()
    column_char_ids = list()
    for column in table["columns"]:
        word_ids_per_column = list()
        char_ids_per_column = list()
        for cell_value in column:
            word_ids = list()
            char_ids_per_cell_value = list()
            words = word_tokenize(cell_value)
            for word in words:
                word_ids.append(lookup_vocab(word_vocab, word))

                char_ids_per_word = list()
                for char in word:
                    char_ids_per_word.append(lookup_vocab(char_vocab, char))
                char_ids_per_cell_value.append(char_ids_per_word)
            char_ids_per_column.append(char_ids_per_cell_value)

            word_ids_per_column.append(word_ids)
        column_word_ids.append(word_ids_per_column)
        column_char_ids.append(char_ids_per_column)

    # Data Type
    column_type_ids = list()
    for column_type in table["column_type"]:
        column_type_ids.append(data_type_vocab[column_type])
    """
    print("Table name: ")
    print(table_name_word_ids)
    print(table_name_char_ids)

    print("=========================")

    print("Column name: ")
    print(column_name_word_ids)
    print(column_name_char_ids)

    print("=========================")

    print("Columns: ")
    print(column_word_ids)
    print(column_char_ids)
    print(len(column_char_ids), len(column_char_ids[0]))
    """
    return {
        "table_name_word_ids": table_name_word_ids,
        "table_name_char_ids": table_name_char_ids,
        "column_name_word_ids": column_name_word_ids,
        "column_name_char_ids": column_name_char_ids,
        "column_word_ids": column_word_ids,
        "column_char_ids": column_char_ids,
        "column_type_ids": column_type_ids
    }


def prepare_tables(file, word_vocab_file, char_vocab_file, data_type_file):
    tables = read_file(file)
    word_vocab = read_vocab(word_vocab_file)
    char_vocab = read_vocab(char_vocab_file)
    data_type_vocab = read_vocab(data_type_file)
    result = list()

    for table in tables:
        table_info = table_lookup_vocab(table, word_vocab, char_vocab, data_type_vocab)
        table_info.update({
            "map_id": table["map_id"]
        })
        result.append(json.dumps(table_info))

    with open(".\\training\\tables.txt", "w") as f:
        f.write('\n'.join(result))


def question_lookup_vocab(question, word_vocab, char_vocab):
    """
    :param question:
    :param word_vocab:
    :param char_vocab:
    :return:
    """
    question_word_ids = list()
    question_char_ids = list()
    tokenized_sentence = question["tokenized_sentence"]

    for word in tokenized_sentence:
        char_ids_per_word = list()
        question_word_ids.append(lookup_vocab(word_vocab, word))
        for char in word:
            char_ids_per_word.append(lookup_vocab(char_vocab, char))
        question_char_ids.append(char_ids_per_word)

    """
    print(tokenized_sentence)
    print(question_word_ids)
    print(question_char_ids)
    """

    return {
        "question_word_ids": question_word_ids,
        "question_char_ids": question_char_ids
    }


def prepare_question(file, word_vocab_file, char_vocab_file):
    """
    :param file:
    :param word_vocab_file:
    :param char_vocab_file:
    :return:
    """
    questions = read_file(file)
    word_vocab = read_vocab(word_vocab_file)
    char_vocab = read_vocab(char_vocab_file)

    result = list()

    for question in questions:
        question_info = question_lookup_vocab(question, word_vocab, char_vocab)
        question_info.update({
            "qid": question["question_id"]["$oid"],
            "ground_truth": question["label_sequence_index"],
            "exact_match": question["table_lookup"],
            "table_map_id": question["table_map_id"]
        })
        result.append(json.dumps(question_info))
    # print(len(result))
    with open(".\\training\\questions.txt", "w") as f:
        f.write('\n'.join(result))


if __name__ == "__main__":
    prepare_tables("..\\data\\training\\tables.txt", ".\\vocab\\word_dict.json", ".\\vocab\\char_dict.json", ".\\vocab\\data_type.json")
    prepare_question("..\\data\\training\\questions.txt", ".\\vocab\\word_dict.json", ".\\vocab\\char_dict.json")

