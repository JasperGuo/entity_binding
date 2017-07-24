# coding=utf8

from nltk import word_tokenize
from prepare_tf_data import read_file, lookup_vocab, read_vocab


def calc_question_words(question_file, word_vocab):
    questions = read_file(question_file)
    word_vocab = read_vocab(word_vocab)

    unknown_word = set()
    all_word = set()

    for question in questions:
        words = question["tokenized_sentence"]
        for word in words:
            word_id = lookup_vocab(word_vocab, word)
            if word_id == 0:
                unknown_word.add(word)
            all_word.add(word)
    print("Total question words: ", len(all_word))
    print("Question unknown words: ", len(unknown_word))
    print("Unknown/all: %f" % (len(unknown_word)/len(all_word)))


def calc_table_words(table_file, word_vocab):
    tables = read_file(table_file)
    word_vocab = read_vocab(word_vocab)

    unknown_word = set()
    all_word = set()

    for table in tables:
        table_words = set(word_tokenize(table["table_name"]))

        for column_name in table["column_name"]:
            column_name_words = set(word_tokenize(column_name))
            table_words |= column_name_words

        for column in table["columns"]:
            for cell_value in column:
                cell_value_words = set(word_tokenize(cell_value))
                table_words |= cell_value_words

        all_word |= table_words

    for word in all_word:
        word_id = lookup_vocab(word_vocab, word)
        if word_id == 0:
            unknown_word.add(word)

    print("Total table words: ", len(all_word))
    print("Table unknown words: ", len(unknown_word))
    print("Unknown/all: %f" % (len(unknown_word)/len(all_word)))


if __name__ == "__main__":
    calc_table_words("..\\data\\training\\tables.txt", ".\\vocab\\word_dict.json")
    calc_question_words("..\\data\\training\\questions.txt", ".\\vocab\\word_dict.json")
