# coding=utf8

import os
import json
from bson import json_util
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from util import read_file


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def lemma(word):
    return lemmatizer.lemmatize(word)


def stem(word):
    return stemmer.stem(word)


def save(file, values):
    with open(file, "w") as f:
        f.write('\n'.join(values))


def process_questions(file):
    questions = read_file(file)
    result = list()
    for question in questions:
        sentence = question["tokenized_sentence"]
        stemmed = list()
        lemmatized = list()
        for word in sentence:
            stemmed.append(stem(word))
            lemmatized.append(lemma(word))
        question["stem"] = stemmed
        question["lemma"] = lemmatized
        result.append(json.dumps(question, default=json_util))

    file_base_name = os.path.basename(file)
    dirname = os.path.dirname(file)
    save_file = os.path.join(dirname, "lemmatized_" + file_base_name)
    save(save_file, result)
    print(len(result))


def process_tables(file):
    tables = read_file(file)
    result = list()
    for table in tables:
        # Table
        lemma_table_name = list()
        stem_table_name = list()
        table_name = word_tokenize(table["table_name"])
        for word in table_name:
            lemma_table_name.append(lemma(word))
            stem_table_name.append(stem(word))

        # Column Name
        lemma_column_name = list()
        stem_column_name = list()
        for cn in table["column_name"]:
            tokenized = word_tokenize(cn)
            l_temp = list()
            s_temp = list()
            for word in tokenized:
                l_temp.append(lemma(word))
                s_temp.append(stem(word))
            lemma_column_name.append(l_temp)
            stem_column_name.append(s_temp)

        # Columns
        lemma_columns = list()
        stem_columns = list()
        for column in table["columns"]:
            l_temp = list()
            s_temp = list()
            for cell in column:
                l = list()
                s = list()
                tokenized = word_tokenize(cell)
                for word in tokenized:
                    l.append(lemma(word))
                    s.append(stem(word))
                l_temp.append(l)
                s_temp.append(s)
            lemma_columns.append(l_temp)
            stem_columns.append(s_temp)
        table.update({
            "lemma_table_name": lemma_table_name,
            "stem_table_name": stem_table_name,
            "lemma_column_name": lemma_column_name,
            "stem_column_name": stem_column_name,
            "lemma_columns": lemma_columns,
            "stem_columns": stem_columns
        })
        result.append(json.dumps(table, default=json_util))

    file_base_name = os.path.basename(file)
    dirname = os.path.dirname(file)
    save_file = os.path.join(dirname, "lemmatized_" + file_base_name)
    save(save_file, result)
    print(len(result))


if __name__ == "__main__":
    process_tables("preprocessed_training_tables.txt")
    process_questions("preprocessed_training_questions.txt")
