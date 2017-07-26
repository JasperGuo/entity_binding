# coding=utf8

import math
import copy
import numpy as np
import json
import random


def read_file(file):
    content = list()
    with open(file) as f:
        for line in f:
            content.append(json.loads(line.strip()))
    return content


def read_vocab(file):
    with open(file, "r") as f:
        return json.load(f)


class VocabManager:
    def __init__(self, file):
        self._vocab = read_vocab(file)
        self._vocab_size = len(list(self._vocab.keys()))

    @property
    def size(self):
        return self._vocab_size


class Batch:
    def __init__(self, **kwargs):
        """
        :param kwargs:
        questions_length:           [batch_size]
        questions_char_ids          [batch_size, max_question_length]
        questions_word_ids          [batch_size, max_question_length]
        tables_name_length          [batch_size]
        tables_name_char_ids        [batch_size, max_table_name_length]
        tables_name_word_ids        [batch_size, max_table_name_length]
        column_name_length          [batch_size, max_column_num]
        column_name_char_ids        [batch_size, max_column_num, max_column_name_length]
        column_name_word_ids        [batch_size, max_column_num, max_column_name_length]
        column_char_ids             [batch_size, max_column_num, max_cell_value_num_per_col, max_cell_value_length]
        column_word_ids             [batch_size, max_column_num, max_cell_value_num_per_col, max_cell_value_length]
        cell_value_length           [batch_size, max_column_num, max_cell_value_num_per_col]
        column_data_type            [batch_size, max_column_num]
        word_character_matrix       [None, max_word_length]
        word_character_length       [None]
        ground_truth                [batch_size, max_question_length]
        exact_match_matrix          [batch_size, max_question_length, None]
        """
        self.questions_length = kwargs["questions_length"]
        self.questions_char_ids = kwargs["questions_char_ids"]
        self.questions_word_ids = kwargs["questions_word_ids"]
        self.tables_name_length = kwargs["tables_name_length"]
        self.tables_name_char_ids = kwargs["tables_name_char_ids"]
        self.tables_name_word_ids = kwargs["tables_name_word_ids"]
        self.column_name_length = kwargs["column_name_length"]
        self.column_name_char_ids = kwargs["column_name_char_ids"]
        self.column_name_word_ids = kwargs["column_name_word_ids"]
        self.column_char_ids = kwargs["column_char_ids"]
        self.column_word_ids = kwargs["column_word_ids"]
        self.cell_value_length = kwargs["cell_value_length"]
        self.column_data_type = kwargs["column_data_type"]
        self.word_character_matrix = kwargs["word_character_matrix"]
        self.word_character_length = kwargs["word_character_length"]
        self.ground_truth = kwargs["ground_truth"]
        self.exact_match_matrix = kwargs["exact_match_matrix"]
        self._learning_rate = 0.0

    @property
    def size(self):
        return len(self.questions_length)

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, rate):
        self._learning_rate = rate

    def _print(self):
        print("Question length: ", self.questions_length, np.array(self.questions_length).shape)
        print("Question char ids: ", self.questions_char_ids, np.array(self.questions_char_ids).shape)
        print("Question word ids: ", self.questions_word_ids, np.array(self.questions_word_ids).shape)
        print("Ground truth: ", self.ground_truth, np.array(self.ground_truth).shape)
        print("Exact match matrix: ", self.exact_match_matrix, "\n", np.array(self.exact_match_matrix).shape)
        print("Table name length: ", self.tables_name_length)
        print("Table name char ids: ", self.tables_name_char_ids, np.array(self.tables_name_char_ids).shape)
        print("Table name word ids: ", self.tables_name_word_ids, np.array(self.tables_name_word_ids).shape)
        print("Column name length: ", self.column_name_length)
        print("Column name char ids: ", self.column_name_char_ids, np.array(self.column_name_char_ids).shape)
        print("Column name word ids: ", self.column_name_word_ids, np.array(self.column_name_word_ids).shape)
        print("Column char ids: ", self.column_char_ids, "\n", np.array(self.column_char_ids).shape)
        print("Column word ids: ", self.column_word_ids, "\n", np.array(self.column_word_ids).shape)
        print("Cell value length: ", self.cell_value_length, "\n", np.array(self.cell_value_length).shape)
        print("Column data type: ", self.column_data_type, "\n", np.array(self.column_data_type).shape)
        print("Word character matrix: ", self.word_character_matrix, np.array(self.word_character_matrix).shape)
        print("Word character length: ", self.word_character_length, np.array(self.word_character_length).shape)


class DataIterator:

    PAD_ID = 0

    @staticmethod
    def get_table_dict(tables):
        table_dict = dict()
        for table in tables:
            table_dict[table["map_id"]] = table
        return table_dict

    def __init__(self, tables_file, questions_file, max_question_length, max_word_length, batch_size):
        """
        :param tables_file:
        :param questions_file:
        :param max_question_length:
        :param max_word_length:
        :param batch_size
        """
        self._cursor = 0
        self._tables = read_file(tables_file)
        self._tables_dict = self.get_table_dict(self._tables)
        self._questions = read_file(questions_file)
        self._max_question_length = max_question_length
        self._max_word_length = max_word_length
        self._batch_size = batch_size
        self._size = len(self._questions)
        self._batch_per_epoch = int(math.floor(self._size / self._batch_size))

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def size(self):
        return len(self._questions)

    @property
    def batch_per_epoch(self):
        return self._batch_per_epoch

    def shuffle(self):
        random.shuffle(self._questions)
        self._cursor = 0

    def _get_word_character_key(self, chars):
        return "::".join([str(c) for c in chars])

    def _get_batch_max_value(self, tables):
        """
        Get batch max value to pad
        :param tables:
        :return:
        """
        max_table_name_length = 0
        max_column_num = 0
        max_column_name_length = 0
        max_cell_value_num_per_col = 0
        max_cell_value_length = 0

        for table in tables:
            # table name
            if len(table["table_name_word_ids"]) > max_table_name_length:
                max_table_name_length = len(table["table_name_word_ids"])

            # column num
            if len(table["column_type_ids"]) > max_column_num:
                max_column_num = len(table["column_type_ids"])

            # max column name length
            for column_name in table["column_name_word_ids"]:
                if len(column_name) > max_column_name_length:
                    max_column_name_length = len(column_name)

            # max cell value num per column
            for column in table["column_word_ids"]:
                if len(column) > max_cell_value_num_per_col:
                    max_cell_value_num_per_col = len(column)

                # max cell value length
                for cell_value in column:
                    if len(cell_value) > max_cell_value_length:
                        max_cell_value_length = len(cell_value)
        return max_table_name_length, max_column_num, max_column_name_length, max_cell_value_num_per_col, max_cell_value_length

    def _pad_chars(self, chars):
        if len(chars) > self._max_word_length:
            _chars = chars[:self._max_word_length]
        elif len(chars) == self._max_word_length:
            _chars = chars
        else:
            _chars = chars + [self.PAD_ID] * (self._max_word_length - len(chars))
        return _chars

    def _reindex_ground_truth(self, ground_truth, table, max_column_num, max_cell_value_num_per_col):
        reindexed = list()
        valid_column_num = len(table["column_type_ids"])
        valid_column_size = [len(c) for c in table["column_word_ids"]]
        valid_index_range = list()
        for idx, size in enumerate(valid_column_size):
            if idx == 0:
                valid_index_range.append(3 + valid_column_num + size - 1)
            else:
                valid_index_range.append(valid_index_range[idx - 1] + size)
        for index in ground_truth:
            if index < (3 + valid_column_num):
                reindexed.append(index)
            else:
                i = 0
                for idx, max_index in enumerate(valid_index_range):
                    if index <= max_index:
                        i = idx
                        break
                if i == 0:
                    diff = index - 3 - valid_column_num
                    valid_idx = 3 + max_column_num + diff
                else:
                    diff = index - valid_index_range[i-1]
                    valid_idx = 3 + max_column_num + max_cell_value_num_per_col * i + diff - 1
                reindexed.append(valid_idx)
        return reindexed

    def _construct_exact_match_matrix(self, exact_match, table, max_column_num, max_cell_value_num_per_col):
        """
        :param exact_match:
        :param table:
        :param max_column_num:
        :param max_cell_value_num_per_col:
        :return:
        """
        matrix = list()
        valid_column_num = len(table["column_type_ids"])
        valid_column_size = [len(c) for c in table["column_word_ids"]]
        valid_index_range = list()
        for idx, size in enumerate(valid_column_size):
            if idx == 0:
                valid_index_range.append(3 + valid_column_num + size - 1)
            else:
                valid_index_range.append(valid_index_range[idx - 1] + size)
        for match in exact_match:
            temp = [0]*(3 + max_column_num + max_cell_value_num_per_col*max_column_num)
            if len(match) == 0:
                matrix.append(temp)
                continue
            for m in match:
                index = m[0]
                i = 0
                for idx, max_index in enumerate(valid_index_range):
                    if index <= max_index:
                        i = idx
                        break
                if i == 0:
                    diff = index - 3 - valid_column_num
                    valid_idx = 3 + max_column_num + diff
                else:
                    diff = index - valid_index_range[i-1]
                    valid_idx = 3 + max_column_num + max_cell_value_num_per_col * i + diff - 1
                temp[valid_idx] = m[1]
            matrix.append(temp)
        return matrix

    def _prepare_batch(self, questions):
        """
        :param questions: [batch_size]
        :return:
        """
        tables = list()

        word_character_dict = dict()
        wid = 1

        # Questions
        # ========================================
        # Shape: [batch_size, max_question_length]
        questions_char_ids = list()
        # Shape: [batch_size, max_question_length]
        questions_word_ids = list()
        # Shape: [batch_size]
        questions_length = list()

        for q in questions:
            tables.append(self._tables_dict[q["table_map_id"]])
            temp = list()
            for word_chars in q["question_char_ids"]:
                # Limit max word length
                _chars = self._pad_chars(word_chars)
                key = self._get_word_character_key(word_chars)
                if key not in word_character_dict:
                    word_character_dict[key] = {
                        "wid": wid,
                        "chars": _chars,
                        "valid_len": len(word_chars)
                    }
                    wid += 1
                temp.append(word_character_dict[key]["wid"])
            questions_length.append(len(temp))
            temp += [self.PAD_ID] * (self._max_question_length - len(temp))
            questions_char_ids.append(temp)

            questions_word_ids.append(q["question_word_ids"] + [self.PAD_ID] * (self._max_question_length - len(q["question_word_ids"])))

        max_table_name_length, max_column_num, max_column_name_length, max_cell_value_num_per_col, max_cell_value_length = self._get_batch_max_value(tables)

        # Ground truth
        # Shape: [batch_size, max_question_length]
        ground_truth = list()
        # Exact match matrix
        # Shape: [batch_size, max_question_length, table_size]
        exact_match_matrix = list()
        for q in questions:
            reindexed_ground_truth = self._reindex_ground_truth(
                q["ground_truth"],
                self._tables_dict[q["table_map_id"]],
                max_column_num=max_column_num,
                max_cell_value_num_per_col=max_cell_value_num_per_col
            )
            ground_truth.append(
                reindexed_ground_truth + [self.PAD_ID] * (self._max_question_length - len(q["ground_truth"])))
            matrix = self._construct_exact_match_matrix(
                exact_match=q["exact_match"],
                table=self._tables_dict[q["table_map_id"]],
                max_column_num=max_column_num,
                max_cell_value_num_per_col=max_cell_value_num_per_col
            )
            exact_match_matrix.append(matrix + [[0.0]*(3 + max_column_num + max_cell_value_num_per_col * max_column_num)]*(self._max_question_length - len(matrix)))

        # Table
        # ==========================================
        # Shape: [batch_size, max_table_name_length]
        tables_name_char_ids = list()
        # Shape: [batch_size, max_table_name_length]
        tables_name_word_ids = list()
        # Shape: [batch_size]
        tables_name_length = list()

        # Shape: [batch_size, max_column_num]
        column_name_length = list()
        # Shape: [batch_size, max_column_num, max_column_name_length]
        column_name_char_ids = list()
        # Shape: [batch_size, max_column_num, max_column_name_length]
        column_name_word_ids = list()

        # Shape: [batch_size, max_column_num, max_cell_value_num_per_col]
        cell_value_length = list()
        # Shape: [batch_size, max_column_num, max_cell_value_num_per_col, max_cell_value_length]
        column_char_ids = list()
        # Shape: [batch_size, max_column_num, max_cell_value_num_per_col, max_cell_value_length]
        column_word_ids = list()

        # Shape: [batch_size, max_column_num]
        column_data_type = list()

        for table in tables:
            # table name
            tname_word_ids = table["table_name_word_ids"]

            tname_char_ids = list()
            for chars in table["table_name_char_ids"]:
                # Limit max word length
                _chars = self._pad_chars(chars)
                key = self._get_word_character_key(chars)
                if key not in word_character_dict:
                    word_character_dict[key] = {
                        "wid": wid,
                        "chars": _chars,
                        "valid_len": len(chars) if len(chars) < self._max_word_length else self._max_word_length
                    }
                    wid += 1
                tname_char_ids.append(word_character_dict[key]["wid"])

            tables_name_length.append(len(tname_word_ids))
            # pad table name
            tname_char_ids += [self.PAD_ID] * (max_table_name_length - len(tname_char_ids))
            tname_word_ids = table["table_name_word_ids"] + [self.PAD_ID] * (max_table_name_length - len(tname_word_ids))

            tables_name_char_ids.append(tname_char_ids)
            tables_name_word_ids.append(tname_word_ids)

            # column name
            cname_char_ids = list()
            for column_name_words in table["column_name_char_ids"]:
                chars_per_column = list()
                for word_chars in column_name_words:
                    # Limit max word length
                    _chars = self._pad_chars(word_chars)
                    key = self._get_word_character_key(word_chars)
                    if key not in word_character_dict:
                        word_character_dict[key] = {
                            "wid": wid,
                            "chars": _chars,
                            "valid_len": len(word_chars) if len(word_chars) < self._max_word_length else self._max_word_length
                        }
                        wid += 1
                    chars_per_column.append(word_character_dict[key]["wid"])
                # pad column name length
                chars_per_column += [self.PAD_ID] * (max_column_name_length - len(chars_per_column))
                cname_char_ids.append(chars_per_column)
            # pad num of column
            cname_char_ids += [[self.PAD_ID] * max_column_name_length] * (max_column_num - len(cname_char_ids))
            cname_word_ids = copy.deepcopy(table["column_name_word_ids"])
            _column_name_length = list()
            for words in cname_word_ids:
                _column_name_length.append(len(words))
                words += [self.PAD_ID] * (max_column_name_length - len(words))
            _column_name_length += [0] * (max_column_num - len(cname_word_ids))
            column_name_length.append(_column_name_length)
            cname_word_ids += [[self.PAD_ID] * max_column_name_length] * (max_column_num - len(cname_word_ids))
            column_name_word_ids.append(cname_word_ids)
            column_name_char_ids.append(cname_char_ids)

            # cell value
            _column_char_ids = list()
            for column in table["column_char_ids"]:
                char_ids_per_column = list()
                for cell_value in column:
                    char_ids_per_cell_value = list()
                    for word_chars in cell_value:
                        # Limit max word length
                        _chars = self._pad_chars(word_chars)
                        key = self._get_word_character_key(word_chars)
                        if key not in word_character_dict:
                            word_character_dict[key] = {
                                "wid": wid,
                                "chars": _chars,
                                "valid_len": len(word_chars) if len(word_chars) < self._max_word_length else self._max_word_length
                            }
                            wid += 1
                        char_ids_per_cell_value.append(word_character_dict[key]["wid"])
                    # pad cell value length
                    char_ids_per_cell_value += [self.PAD_ID] * (max_cell_value_length - len(char_ids_per_cell_value))
                    char_ids_per_column.append(char_ids_per_cell_value)
                # pad num of cell values per column
                char_ids_per_column += [[self.PAD_ID] * max_cell_value_length] * (max_cell_value_num_per_col - len(char_ids_per_column))
                _column_char_ids.append(char_ids_per_column)
            # pad num of column
            _column_char_ids += [[[self.PAD_ID] * max_cell_value_length] * max_cell_value_num_per_col] * (max_column_num - len(_column_char_ids))
            column_char_ids.append(_column_char_ids)

            _column_word_ids = copy.deepcopy(table["column_word_ids"])
            _cell_value_length = list()
            for column in _column_word_ids:
                _cell_value_length_per_column = list()
                for words in column:
                    _cell_value_length_per_column.append(len(words))
                    words += [self.PAD_ID] * (max_cell_value_length - len(words))
                _cell_value_length_per_column += [0] * (max_cell_value_num_per_col - len(column))
                _cell_value_length.append(_cell_value_length_per_column)
                column += [[self.PAD_ID] * max_cell_value_length] * (max_cell_value_num_per_col - len(column))
            _cell_value_length += [[0] * max_cell_value_num_per_col] * (max_column_num - len(_cell_value_length))
            cell_value_length.append(_cell_value_length)
            _column_word_ids += [[[self.PAD_ID] * max_cell_value_length] * max_cell_value_num_per_col] * (max_column_num - len(_column_word_ids))
            column_word_ids.append(_column_word_ids)

            # Column data type
            column_data_type.append(table["column_type_ids"] + [self.PAD_ID] * (max_column_num - len(table["column_type_ids"])))

        # Construct Word Character Matrix
        # ==========================================
        matrix = sorted([(v["wid"], v["valid_len"], v["chars"]) for k, v in word_character_dict.items()],
                        key=lambda x: x[0])
        # Shape: [None, max_word_length]
        word_character_matrix = list()
        word_character_length = list()
        for m in matrix:
            word_character_matrix.append(m[2])
            word_character_length.append(m[1])

        return Batch(
            questions_length=questions_length,
            questions_word_ids=questions_word_ids,
            questions_char_ids=questions_char_ids,
            ground_truth=ground_truth,
            exact_match_matrix=exact_match_matrix,
            tables_name_length=tables_name_length,
            tables_name_char_ids=tables_name_char_ids,
            tables_name_word_ids=tables_name_word_ids,
            column_name_length=column_name_length,
            column_name_char_ids=column_name_char_ids,
            column_name_word_ids=column_name_word_ids,
            column_char_ids=column_char_ids,
            column_word_ids=column_word_ids,
            cell_value_length=cell_value_length,
            column_data_type=column_data_type,
            word_character_matrix=word_character_matrix,
            word_character_length=word_character_length
        )

    def get_batch(self):

        """
        if self._cursor + self._batch_per_epoch > self._size:
            raise IndexError("Index Error")
        """
        questions = self._questions[self._cursor:self._cursor+self._batch_size]
        self._cursor += self._batch_size
        return self._prepare_batch(questions)


if __name__ == "__main__":
    data_iterator = DataIterator(
        "..\\..\\tf_data\\training\\tables.txt",
        "..\\..\\tf_data\\training\\questions.txt",
        21,
        20,
        1
    )
    data_iterator.shuffle()
    batch = data_iterator.get_batch()
    batch._print()

