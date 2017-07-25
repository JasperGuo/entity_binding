# coding=utf8

import tensorflow as tf


def get_last_relevant(output, length):
    slices = list()
    for idx, l in enumerate(tf.unstack(length)):
        last = tf.slice(output, begin=[idx, l - 1, 0], size=[1, 1, -1])
        slices.append(last)
    lasts = tf.concat(slices, 0)
    return lasts


class Model:

    def __init__(
            self,
            opts,
            word_vocab_size,
            char_vocab_size,
            column_data_type_num,
            pretrain_word_embedding,
            is_test=False
    ):
        self._is_test = is_test
        self._batch_size = opts["batch_size"]

        self._word_embedding_dim = opts["word_embedding_dim"]
        self._char_embedding_dim = opts["char_embedding_dim"]
        self._data_type_embedding_dim = opts["data_type_embedding_dim"]
        self._gradient_clip = opts["gradient_clip"]
        self._dropout = opts["dropout"]
        self._max_question_length = opts["max_question_length"]
        self._max_word_length = opts["max_word_length"]
        self._char_rnn_encoder_hidden_dim = opts["character_rnn_encoder_hidden_dim"]
        self._char_rnn_encoder_layer = opts["character_rnn_encoder_layer"]
        self._question_rnn_encoder_hidden_dim = opts["question_rnn_encoder_hidden_dim"]
        self._question_rnn_encoder_layer = opts["question_rnn_encoder_layer"]
        self._combined_embedding_dim = opts["combined_embedding_dim"]
        self._word_vocab_size = word_vocab_size
        self._char_vocab_size = char_vocab_size
        self._column_data_type_num = column_data_type_num
        self._pretrain_word_embedding = pretrain_word_embedding

        if self._is_test:
            self._build_test_graph()
        else:
            self._build_train_graph()

    def _build_input_nodes(self):
        with tf.name_scope("model_placeholder"):
            # Questions
            self._questions_length = tf.placeholder(
                tf.int32,
                [self._batch_size],
                name="questions_length"
            )
            self._questions_char_ids = tf.placeholder(
                tf.int32,
                [self._batch_size, self._max_question_length],
                name="questions_char_ids"
            )
            self._questions_word_ids = tf.placeholder(
                tf.int32,
                [self._batch_size, self._max_question_length],
                name="questions_word_ids"
            )

            # Table Name
            # batch_size
            # None: table_name_length
            self._tables_name_length = tf.placeholder(
                tf.int32,
                [self._batch_size],
                name="tables_name_length"
            )
            self._tables_name_char_ids = tf.placeholder(
                tf.int32,
                [self._batch_size, None],
                name="tables_name_char_ids"
            )
            self._tables_name_word_ids = tf.placeholder(
                tf.int32,
                [self._batch_size, None],
                name="tables_name_word_ids"
            )

            # Column Name
            # batch_size
            # None: column_num
            # None: column_name_length
            self._column_name_length = tf.placeholder(
                tf.int32,
                [self._batch_size, None],
                name="column_name_length"
            )
            self._column_name_char_ids = tf.placeholder(
                tf.int32,
                [self._batch_size, None, None],
                name="column_name_char_ids"
            )
            self._column_name_word_ids = tf.placeholder(
                tf.int32,
                [self._batch_size, None, None],
                name="column_name_word_ids"
            )

            # Cell Value
            # batch_size
            # None: column_num
            # None: cell_value_num_per_col
            # None: cell_value_length
            self._cell_value_char_ids = tf.placeholder(
                tf.int32,
                [self._batch_size, None, None, None],
                name="cell_value_char_ids"
            )
            self._cell_value_word_ids = tf.placeholder(
                tf.int32,
                [self._batch_size, None, None, None],
                name="cell_value_word_ids"
            )
            self._cell_value_length = tf.placeholder(
                tf.int32,
                [self._batch_size, None, None],
                name="cell_value_length"
            )

            # Column data type
            # batch_size
            # None: column_num
            self._column_data_type = tf.placeholder(
                tf.int32,
                [self._batch_size, None],
                name="column_data_type"
            )

            # Word-Character matrix
            # None: num of words
            self._word_character_matrix = tf.placeholder(
                tf.int32,
                [None, self._max_word_length],
                name="word_character_matrix"
            )
            self._word_character_length = tf.placeholder(
                tf.int32,
                [None],
                name="word_character_length"
            )

            self._ground_truth = tf.placeholder(
                tf.int32,
                [self._batch_size, self._max_question_length],
                name="ground_truth"
            )

            # Exact match matrix
            # None: size of table
            self._exact_match_matrix = tf.placeholder(
                tf.float32,
                [self._batch_size, self._max_question_length, None],
                name="exact_match_matrix"
            )

            # Dropout
            self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _set_dynamic_value(self):
        self._max_column_num = tf.shape(self._column_name_length)[1]
        self._max_table_name_length = tf.shape(self._tables_name_length)[1]
        self._max_cell_value_num_per_col = tf.shape(self._cell_value_length)[2]
        self._max_cell_value_length = tf.shape(self._cell_value_length)[3]

    def _build_embedding(self):
        with tf.variable_scope("word_embedding_layer"):
            word_pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._word_embedding_dim], dtype=tf.float32),
                name="word_pad_embedding",
                trainable=False
            )
            word_embedding = tf.get_variable(
                name="word_embedding",
                trainable=False,
                shape=[self._word_vocab_size, self._word_embedding_dim],
                dtype=tf.float32,
                initializer=tf.constant_initializer(self._pretrain_word_embedding)
            )
            word_embedding = tf.concat(values=[word_pad_embedding, word_embedding], axis=0)

        with tf.variable_scope("character_embedding_layer"):
            char_pad_embedding = tf.get_variable(
                initializer=tf.zeros([1, self._char_embedding_dim], dtype=tf.float32),
                name="char_pad_embedding",
                trainable=False
            )
            char_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._char_vocab_size - 1, self._char_embedding_dim],
                    stddev=0.5
                ),
                name="char_embedding"
            )
            char_embedding = tf.concat(values=[char_pad_embedding, char_embedding], axis=0)

        with tf.variable_scope("special_tag_embedding_layer"):
            # PAT & LIT
            special_tag_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [2, self._char_embedding_dim],
                    stddev=0.5
                ),
                name="special_tag_embedding"
            )

        with tf.variable_scope("column_data_type_embedding_layer"):
            column_data_type_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._column_data_type_num, self._data_type_embedding_dim],
                    stddev=0.5
                ),
                name="column_data_type_embedding"
            )

        return word_embedding, char_embedding, column_data_type_embedding, special_tag_embedding

    def _encode_word(self, embedded_character, word_length):
        """
        BiRNN to encode character based word embedding
        :param embedded_character:  [None, max_word_length, character_embedding_size]
        :param word_length:         [None]
        :return:
            [None, char_rnn_encoder_hidden_dim*2]
        """
        with tf.variable_scope("character_encoder"):
            with tf.variable_scope("fw_cell"):
                fw_cell = tf.contrib.rnn.GRUCell(
                    num_units=self._char_rnn_encoder_hidden_dim
                )
                fw_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=fw_cell,
                    input_keep_prob=self._dropout_keep_prob,
                    output_keep_prob=self._dropout_keep_prob
                )
                fw_cell = tf.contrib.rnn.MultiRNNCell(
                    [fw_cell] * self._char_rnn_encoder_layer
                )
            with tf.variable_scope("bw_cell"):
                bw_cell = tf.contrib.rnn.GRUCell(
                    num_units=self._char_rnn_encoder_hidden_dim
                )
                bw_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=bw_cell,
                    input_keep_prob=self._dropout_keep_prob,
                    output_keep_prob=self._dropout_keep_prob
                )
                bw_cell = tf.contrib.rnn.MultiRNNCell(
                    [bw_cell] * self._char_rnn_encoder_layer
                )
        with tf.name_scope("encode_character") as f:
            (output_fw, output_bw), (output_states_fw, output_states_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=embedded_character,
                sequence_length=word_length
            )

            _output_fw = get_last_relevant(output=output_fw, length=word_length)
            _output_bw = get_last_relevant(output=output_bw, length=word_length)

            return tf.concat(values=[_output_fw, _output_bw], axis=-1)

    def _combine_word_embedding_and_character_based_embedding(self, word_embedded, character_embedded):
        """
        :param word_embedded:
        :param character_embedded:
        :return:
        """
        with tf.name_scope("combine_word_and_character_embedding"):
            return tf.layers.dense(
                inputs=tf.concat(values=[word_embedded, character_embedded], axis=1),
                units=self._combined_embedding_dim,
                activation=tf.nn.relu,
                name="combine_embedding",
                reuse=True
            )

    def _encode_question(self, character_based_word_embedding, word_embedding):
        """
        :param character_based_word_embedding:
        :param word_embedding:
        :return:
            [batch_size, max_question_length, word_embedding_dim]
        """
        char_embedded_question = tf.nn.embedding_lookup(
            params=character_based_word_embedding,
            ids=self._questions_char_ids
        )
        word_embedded_question = tf.nn.embedding_lookup(
            params=word_embedding,
            ids=self._questions_word_ids
        )
        # Shape: [batch_size, max_question_length, combined_embedding_dim]
        combined_embedded = self._combine_word_embedding_and_character_based_embedding(
            word_embedded=word_embedded_question,
            character_embedded=char_embedded_question
        )

        with tf.variable_scope("question_encoder"):
            with tf.variable_scope("fw_cell"):
                fw_cell = tf.contrib.rnn.GRUCell(
                    num_units=self._question_rnn_encoder_hidden_dim
                )
                fw_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=fw_cell,
                    input_keep_prob=self._dropout_keep_prob,
                    output_keep_prob=self._dropout_keep_prob
                )
                fw_cell = tf.contrib.rnn.MultiRNNCell(
                    [fw_cell] * self._question_rnn_encoder_layer
                )
            with tf.variable_scope("bw_cell"):
                bw_cell = tf.contrib.rnn.GRUCell(
                    num_units=self._question_rnn_encoder_hidden_dim
                )
                bw_cell = tf.contrib.rnn.DropoutWrapper(
                    cell=bw_cell,
                    input_keep_prob=self._dropout_keep_prob,
                    output_keep_prob=self._dropout_keep_prob
                )
                bw_cell = tf.contrib.rnn.MultiRNNCell(
                    [bw_cell] * self._question_rnn_encoder_layer
                )
        with tf.name_scope("encode_character") as f:
            (output_fw, output_bw), (output_states_fw, output_states_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw_cell,
                cell_bw=bw_cell,
                inputs=combined_embedded,
                sequence_length=self._questions_length
            )

            # Shape: [batch_size, max_question_length, question_rnn_encoder_hidden_dim*2]
            question_rnn_outputs = tf.concat(values=[output_fw, output_bw], axis=-1)

            return tf.layers.dense(
                inputs=tf.concat(values=[question_rnn_outputs, combined_embedded], axis=1),
                units=self._word_embedding_dim,
                activation=tf.nn.relu
            )

    def _encode_table(self):
        pass

    def _build_test_graph(self):
        pass

    def _build_train_graph(self):
        self._build_input_nodes()
        self._set_dynamic_value()
        word_embedding, character_embedding, data_type_embedding, special_tag_embedding = self._build_embedding()

        embedded_character = tf.nn.embedding_lookup(
            params=character_embedding,
            ids=self._word_character_matrix
        )
        # Shape: [None, character_embedding_dim]
        character_based_word_embedding = self._encode_word(
            embedded_character=embedded_character,
            word_length=self._word_character_length
        )

        # Shape: [batch_size, max_question_length, word_embedding_dim]
        question_representation = self._encode_question(
            character_based_word_embedding=character_based_word_embedding,
            word_embedding=word_embedding
        )
