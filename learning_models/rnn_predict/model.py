# coding=utf8

import tensorflow as tf


def get_last_relevant(output, length):
    slices = list()
    for idx, l in enumerate(tf.unstack(length)):
        last = tf.slice(output, begin=[idx, l - 1, 0], size=[1, 1, -1])
        slices.append(last)
    lasts = tf.concat(slices, 0)
    return lasts


def softmax_with_mask(tensor, mask):
    """
    Calculate Softmax with mask
    :param tensor: [shape1, shape2]
    :param mask:   [shape1, shape2]
    :return:
    """
    exp_tensor = tf.exp(tensor)
    masked_exp_tensor = tf.multiply(exp_tensor, mask)
    total = tf.reshape(
        tf.reduce_sum(masked_exp_tensor, axis=1),
        shape=[-1, 1]
    )
    return tf.div(masked_exp_tensor, total)


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
        self._scoring_weight_dim = opts["scoring_weight_dim"]
        self._cell_value_encoder_layer_1_dim = opts["cell_value_encoder_layer_1_dim"]
        self._cell_value_encoder_layer_2_dim = opts["cell_value_encoder_layer_2_dim"]
        self._predict_rnn_decoder_hidden_dim = opts["predict_rnn_decoder_hidden_dim"]

        self._table_extra_transform_dim = opts["table_name_and_column_name_transform_dim"]

        self._word_vocab_size = word_vocab_size
        self._char_vocab_size = char_vocab_size
        self._column_data_type_num = column_data_type_num
        self._pretrain_word_embedding = pretrain_word_embedding

        if self._is_test:
            self._batch_size = opts["test_batch_size"]
        else:
            self._batch_size = opts["batch_size"]


        self._build_graph()

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

            self._dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            if not self._is_test:
                # Dropout
                self._learning_rate = tf.placeholder(tf.float32, name="learning_rate")

    def _set_dynamic_value(self):
        self._max_column_num = tf.shape(self._column_name_length)[1]
        self._max_table_name_length = tf.shape(self._tables_name_char_ids)[1]
        self._max_column_name_length = tf.shape(self._column_name_char_ids)[2]
        self._max_cell_value_num_per_col = tf.shape(self._cell_value_length)[2]
        self._max_cell_value_length = tf.shape(self._cell_value_word_ids)[3]

        # LIT + PAT + table_name + num_of_columns + cell_values
        self._table_size = 2 + 1 + self._max_column_num + self._max_column_num * self._max_cell_value_num_per_col

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
                    [self._char_vocab_size, self._char_embedding_dim],
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

        with tf.variable_scope("go_tag_embedding_layer"):
            # GO
            go_tag_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._char_embedding_dim],
                    stddev=0.5
                ),
                name="go_tag_embedding"
            )

        with tf.variable_scope("column_data_type_embedding_layer"):
            column_data_type_embedding = tf.get_variable(
                initializer=tf.truncated_normal(
                    [self._column_data_type_num, self._data_type_embedding_dim],
                    stddev=0.5
                ),
                name="column_data_type_embedding"
            )

        return word_embedding, char_embedding, column_data_type_embedding, special_tag_embedding, go_tag_embedding

    def _encode_word(self, embedded_character, word_length):
        """
        BiRNN to encode character based word embedding
        :param embedded_character:  [None, max_word_length, character_embedding_size]
        :param word_length:         [None]
        :return:
            [None, char_rnn_encoder_hidden_dim*2]
        """

        with tf.variable_scope("word_character_pad_embedding"):
            pad_embedding = tf.get_variable(
                initializer=tf.zeros_initializer,
                shape=[1, self._char_rnn_encoder_hidden_dim*2],
                name="pad"
            )

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
            with tf.variable_scope("BiGRU_encode_character"):
                (output_fw, output_bw), (output_states_fw, output_states_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell,
                    cell_bw=bw_cell,
                    inputs=embedded_character,
                    sequence_length=word_length,
                    dtype=tf.float32
                )

            # dangerous implementation, fail when there are more than one layers
            _output_fw = output_states_fw[0]
            _output_bw = output_states_bw[0]

            return tf.concat(
                values=[
                    pad_embedding,
                    tf.concat(values=[_output_fw, _output_bw], axis=-1),
                ],
                axis=0
            )

    def _initialize_combine_embedding_layer(self):
        with tf.variable_scope("combine_embedding_layer"):
            W = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[
                    self._word_embedding_dim + self._char_embedding_dim,
                    self._combined_embedding_dim
                ],
                name="weight"
            )
            b = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._combined_embedding_dim],
                name="bias"
            )

            return {
                "W": W,
                "b": b
            }

    def _combine_embedding(self, params, word_embedded, character_embedded):
        """
        :param params: {"W":W, "b": b}
        :param word_embedded:
        :param character_embedded:
        :return:
        """
        with tf.name_scope("combine_word_and_character_embedding"):
            dropout_layer = tf.nn.dropout(
                x=tf.concat(values=[word_embedded, character_embedded], axis=-1),
                keep_prob=self._dropout_keep_prob
            )
            return tf.nn.relu(
                tf.add(
                    tf.matmul(
                        a=dropout_layer,
                        b=params["W"]
                    ),
                    params["b"]
                )
            )

    def _encode_question(self, character_based_word_embedding, word_embedding, combine_layer_params):
        """
        :param character_based_word_embedding:
        :param word_embedding:
        :param combine_layer_params
        :return:
            [batch_size, max_question_length, table_extra_transform_dim]
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
        combined_embedded = tf.reshape(
            self._combine_embedding(
                params=combine_layer_params,
                word_embedded=tf.reshape(word_embedded_question, shape=[-1, self._word_embedding_dim]),
                character_embedded=tf.reshape(char_embedded_question, shape=[-1, self._char_embedding_dim])
            ),
            shape=[self._batch_size, self._max_question_length, self._combined_embedding_dim]
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
        with tf.name_scope("encode_question"):
            with tf.variable_scope("BiGRU_encode_question"):
                (output_fw, output_bw), (output_states_fw, output_states_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=fw_cell,
                    cell_bw=bw_cell,
                    inputs=combined_embedded,
                    sequence_length=self._questions_length,
                    dtype=tf.float32
                )

            # Shape: [batch_size, max_question_length, question_rnn_encoder_hidden_dim*2]
            question_rnn_outputs = tf.concat(values=[output_fw, output_bw], axis=-1)

            with tf.variable_scope("transform"):
                return tf.reshape(
                    tf.layers.dense(
                        inputs=question_rnn_outputs,
                        units=self._table_extra_transform_dim,
                        name="highway_transform_layer_2"
                    ),
                    shape=[self._batch_size, self._max_question_length, self._table_extra_transform_dim]
                ), combined_embedded

    def _encode_table_name(self, character_based_word_embedding, word_embedding, combine_layer_params):
        """
        Encode table name
        :param character_based_word_embedding:
        :param word_embedding:
        :param combine_layer_params
        :return:
            [batch_size, combined_embedding_dim]
        """
        char_embedded = tf.nn.embedding_lookup(
            params=character_based_word_embedding,
            ids=self._tables_name_char_ids
        )
        word_embedded = tf.nn.embedding_lookup(
            params=word_embedding,
            ids=self._tables_name_word_ids
        )
        # Shape: [batch_size, combined_embedding_dim]
        combined_embedded = tf.reduce_sum(
            tf.reshape(
                self._combine_embedding(
                    params=combine_layer_params,
                    word_embedded=tf.reshape(word_embedded, shape=[-1, self._word_embedding_dim]),
                    character_embedded=tf.reshape(char_embedded, shape=[-1, self._word_embedding_dim])
                ),
                shape=[self._batch_size, self._max_table_name_length, self._combined_embedding_dim]
            ),
            axis=1
        )
        return combined_embedded

    def _encode_column_name(self, character_based_word_embedding, word_embedding, combine_layer_params):
        """
        Encode column name
        :param character_based_word_embedding:
        :param word_embedding:
        :param combine_layer_params
        :return:
            [batch_size, max_column_num, combined_embedding_dim]
        """
        char_embedded = tf.nn.embedding_lookup(
            params=character_based_word_embedding,
            ids=self._column_name_char_ids
        )
        word_embedded = tf.nn.embedding_lookup(
            params=word_embedding,
            ids=self._column_name_word_ids
        )
        flatten_char_embedded = tf.reshape(char_embedded, shape=[-1, self._char_embedding_dim])
        flatten_word_embedded = tf.reshape(word_embedded, shape=[-1, self._word_embedding_dim])

        # Shape: [batch_size, max_column_num, combined_embedding_dim]
        combined_embedded = tf.reduce_sum(
            tf.reshape(
                self._combine_embedding(
                    params=combine_layer_params,
                    word_embedded=flatten_char_embedded,
                    character_embedded=flatten_word_embedded
                ),
                shape=[self._batch_size, self._max_column_num, self._max_column_name_length,
                       self._combined_embedding_dim]
            ),
            axis=2
        )
        return combined_embedded

    def _encode_cell_value(self,
                           character_based_word_embedding,
                           word_embedding,
                           data_type_embedding,
                           combine_layer_params,
                           embedded_column_name,
                           ):
        # TODO: add a mask
        """
        Encode Cell Value,  Concatenate data type embedding and column name embedding
        :param character_based_word_embedding:
        :param word_embedding:
        :param combine_layer_params:
        :param embedded_column_name:
        :param data_type_embedding
        :return:
            [batch_size, max_column_num, max_cell_value_per_col, cell_value_encoder_layer_2_dim]
        """
        # Shape: [batch_size, max_column_num, max_cell_value_per_col, max_cell_value_length, word_embedding_dim]
        char_embedded = tf.nn.embedding_lookup(
            params=character_based_word_embedding,
            ids=self._cell_value_char_ids
        )
        # Shape: [batch_size, max_column_num, max_cell_value_per_col, max_cell_value_length, word_embedding_dim]
        word_embedded = tf.nn.embedding_lookup(
            params=word_embedding,
            ids=self._cell_value_word_ids
        )
        flatten_char_embedded = tf.reshape(char_embedded, shape=[-1, self._char_embedding_dim])
        flatten_word_embedded = tf.reshape(word_embedded, shape=[-1, self._word_embedding_dim])

        # Shape: [batch_size, max_column_num, max_cell_value_num_per_col, combined_embedding_dim]
        combined_embedded = tf.reduce_sum(
            tf.reshape(
                self._combine_embedding(
                    params=combine_layer_params,
                    word_embedded=flatten_char_embedded,
                    character_embedded=flatten_word_embedded
                ),
                shape=[
                    self._batch_size,
                    self._max_column_num,
                    self._max_cell_value_num_per_col,
                    self._max_cell_value_length,
                    self._combined_embedding_dim
                ]
            ),
            axis=3
        )

        # Shape: [batch_size, max_column_num, data_type_embedding_dim]
        embedded_column_data_type = tf.nn.embedding_lookup(
            params=data_type_embedding,
            ids=self._column_data_type
        )

        # Shape: [batch_size, max_column_num, data_type_embedding_dim + combined_embedding_Dim]
        column_name_data_type_embedding = tf.concat(
            values=[embedded_column_name, embedded_column_data_type],
            axis=-1
        )

        with tf.variable_scope("cell_value_encoder"):
            rich_embedding = tf.reshape(
                tf.concat(
                    values=[
                        tf.tile(tf.expand_dims(column_name_data_type_embedding, axis=2),
                                multiples=[1, 1, self._max_cell_value_num_per_col, 1]),
                        combined_embedded
                    ],
                    axis=-1
                ),
                shape=[-1, self._data_type_embedding_dim + self._combined_embedding_dim + self._combined_embedding_dim]
            )
            dropout_layer = tf.nn.dropout(
                x=rich_embedding,
                keep_prob=self._dropout_keep_prob
            )
            layer_1 = tf.layers.dense(
                inputs=dropout_layer,
                units=self._cell_value_encoder_layer_1_dim,
                activation=tf.nn.relu,
                name="encode_cell_value_layer_1"
            )
            layer_2 = tf.layers.dense(
                inputs=layer_1,
                units=self._cell_value_encoder_layer_2_dim,
                name="encode_cell_value_layer_2"
            )

            return tf.reshape(
                layer_2,
                shape=[self._batch_size, self._max_column_num, self._max_cell_value_num_per_col,
                       self._cell_value_encoder_layer_2_dim]
            )

    def _transform_column_name_and_table_name(self, table_name_representation, column_name_representation):
        """
        :param table_name_representation:   [batch_size, combined_embedding_dim]
        :param column_name_representation:  [batch_size, max_column_num, combined_embedding_dim]
        :return:
            table_name:     [batch_size, table_extra_transform_dim],
            column_name:    [batch_size, max_column_num, table_extra_transform_dim]
        """
        with tf.variable_scope("transform_column_name_and_table_name"):
            # Shape: [batch_size + batch_size * max_column_num, table_extra_transform_dim]
            transformed = tf.layers.dense(
                inputs=tf.concat(
                    values=[
                        table_name_representation,
                        tf.reshape(
                            column_name_representation,
                            shape=[-1, self._combined_embedding_dim]
                        )
                    ],
                    axis=0
                ),
                units=self._table_extra_transform_dim,
                activation=tf.nn.relu
            )

            transformed_table_name_representation = tf.slice(
                transformed,
                begin=[0, 0],
                size=[self._batch_size, self._table_extra_transform_dim]
            )

            transformed_column_name_representation = tf.reshape(
                tf.slice(
                    transformed,
                    begin=[self._batch_size, 0],
                    size=[-1, self._table_extra_transform_dim]
                ),
                shape=[self._batch_size, self._max_column_num, self._table_extra_transform_dim]
            )

            return transformed_table_name_representation, transformed_column_name_representation

    def _transform_special_tag(self, special_tag_embedding):
        """
        Transform special tag
        :param special_tag_embedding:
        :return:
            [2, table_extra_transform_dim]
        """
        with tf.variable_scope("transform_special_tag_embedding"):
            return tf.layers.dense(
                inputs=special_tag_embedding,
                units=self._table_extra_transform_dim,
                activation=tf.nn.relu
            )

    def _flatten_table(self, table_name_representation, column_name_representation, cell_value_representation, special_tag):
        """
        Flatten table
        :param table_name_representation:   [batch_size, table_extra_transform_dim]
        :param column_name_representation:  [batch_size, max_column_num, table_extra_transform_dim]
        :param cell_value_representation:   [batch_size, max_column_num, max_cell_value_per_col, table_extra_transform_dim]
        :param special_tag:                 [2, table_extra_transform_dim]
        :return:
            [batch_size, table_size, table_extra_transform_dim]
        """
        # Shape: [batch_size, 2, table_extra_transform_dim]
        expanded_special_tag = tf.tile(
            tf.expand_dims(
                special_tag,
                axis=0
            ),
            multiples=[self._batch_size, 1, 1]
        )
        # Shape: [batch_size, table_size, table_extra_transform_dim]
        table_representation = tf.concat(
            values=[
                expanded_special_tag,
                tf.expand_dims(
                    table_name_representation,
                    axis=1
                ),
                column_name_representation,
                tf.reshape(
                    cell_value_representation,
                    shape=[self._batch_size, self._max_column_num*self._max_cell_value_num_per_col, self._table_extra_transform_dim]
                )
            ],
            axis=1
        )
        return table_representation

    def _build_predict_rnn_decoder(self):
        with tf.variable_scope("cell"):
            rnn_cell = tf.contrib.rnn.GRUCell(
                num_units=self._predict_rnn_decoder_hidden_dim
            )
            rnn_cell = tf.contrib.rnn.DropoutWrapper(
                cell=rnn_cell,
                input_keep_prob=self._dropout_keep_prob,
                output_keep_prob=self._dropout_keep_prob
            )
            rnn_cell = tf.contrib.rnn.MultiRNNCell(
                [rnn_cell] * self._question_rnn_encoder_layer
            )
        return rnn_cell

    def _initialize_table_attention_layer(self):
        with tf.variable_scope("table_attention_layer"):
            W = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[
                    self._predict_rnn_decoder_hidden_dim + self._combined_embedding_dim,
                    self._table_extra_transform_dim
                ],
                name="weight"
            )
            b = tf.get_variable(
                initializer=tf.zeros_initializer(),
                shape=[self._combined_embedding_dim],
                name="bias"
            )
            output_w = tf.get_variable(
                initializer=tf.contrib.layers.xavier_initializer(),
                shape=[
                    self._table_extra_transform_dim,
                    self._table_extra_transform_dim
                ],
                name="output_W"
            )
            return {
                "W": W,
                "b": b,
                "output_W": output_w
            }

    def _calc_table_attention(self, table_representation, question_raw_embedding, weights, rnn_outputs):
        """
        :param table_representation: [batch_size, table_size, table_extra_transform_dim]
        :param question_raw_embedding: [batch_size, max_question_size, combined_embedding_dim]
        :param weights:              {W, b, output_W}
        :param rnn_outputs:          [batch_size, max_question_size, predict_rnn_decoder_hidden_dim]
        :return:
        """
        with tf.name_scope("table_attention"):
            # Shape: [batch_size, max_question_size, table_extra_transform_dim]
            word_embedding_awared_rnn_outputs = tf.reshape(
                tf.matmul(
                    tf.nn.relu(
                        tf.add(
                            tf.matmul(
                                tf.reshape(
                                    tf.concat(values=[question_raw_embedding, rnn_outputs], axis=-1),
                                    shape=[-1, self._combined_embedding_dim + self._predict_rnn_decoder_hidden_dim]
                                ),
                                weights["W"]
                            ),
                            weights["b"]
                        )
                    ),
                    weights["output_W"]
                ),
                shape=[self._batch_size, self._max_question_length, self._table_extra_transform_dim]
            )
            # Shape: [batch_size, table_size*max_question_length, table_extra_transform_dim]
            expanded_table_representation = tf.tile(
                table_representation,
                multiples=[1, self._max_question_length, 1]
            )
            # Shape: [batch_size, max_question_length*self.table_size, table_extra_transform_dim]
            expanded_question_representation = tf.reshape(
                tf.tile(
                    tf.expand_dims(
                        word_embedding_awared_rnn_outputs,
                        axis=2
                    ),
                    multiples=[1, 1, self._table_size, 1]
                ),
                shape=[self._batch_size, self._max_question_length * self._table_size, self._table_extra_transform_dim]
            )

            # Shape: [batch_size, question_length, table_size]
            return tf.nn.softmax(
                tf.reshape(
                    tf.reduce_sum(
                        tf.multiply(
                            expanded_question_representation,
                            expanded_table_representation
                        ),
                        axis=-1
                    ),
                    shape=[self._batch_size, self._max_question_length, self._table_size]
                ),
                dim=-1
            )

    def _train_predict_tag(self, table_representation, question_representation, question_raw_embedding, go_tag_embedding):
        """
        Predict tag in train phase
        :param table_representation:        [batch_size, table_size, table_extra_transform_dim]
        :param question_representation:     [batch_size, max_question_length, table_extra_transform_dim]
        :param question_raw_embedding:      [batch_size, max_question_length, combined_embedding_dim]
        :param go_tag_embedding:            [table_extra_transform_dim]
        :return:
            [batch_size, max_question_length, table_size]
        """
        with tf.name_scope("train_predict_tag"):

            decoder_cell = self._build_predict_rnn_decoder()

            # Shape: [batch_size * max_question_length, table_extra_transform_dim]
            ground_truth_embedding_index = tf.reshape(
                self._calc_ground_truth_embedding_index(),
                shape=[-1, 2]
            )
            # Shape: [batch_size, max_question_length - 1, table_extra_transform_dim]
            rnn_table_embedding = tf.slice(
                tf.reshape(
                    tf.gather_nd(params=table_representation, indices=ground_truth_embedding_index),
                    shape=[self._batch_size, self._max_question_length, self._table_extra_transform_dim]
                ),
                begin=[0, 0, 0],
                size=[self._batch_size, self._max_question_length - 1, self._table_extra_transform_dim]
            )
            # Shape: [batch_size, 1, table_extra_transform_dim]
            expanded_go_tab_embedding = tf.expand_dims(
                tf.tile(
                    tf.expand_dims(go_tag_embedding, axis=0),
                    multiples=[self._batch_size, 1]
                ),
                axis=1
            )
            # Shape: [batch_size, max_question_length, table_extra_transform_dim]
            rnn_table_embedding = tf.concat([expanded_go_tab_embedding, rnn_table_embedding], axis=1)

            # Shape: [batch_size, max_question_length, table_extra_transform_dim*2]
            decoder_input = tf.concat([question_representation, rnn_table_embedding], axis=-1)

            with tf.variable_scope("decode_tag"):
                # Shape: [batch_size, max_question_length, predict_rnn_decoder_hidden_dim]
                decoder_outputs, decoder_states = tf.nn.dynamic_rnn(
                    cell=decoder_cell,
                    inputs=decoder_input,
                    sequence_length=self._questions_length,
                    dtype=tf.float32
                )
            table_attention_weights = self._initialize_table_attention_layer()
            scores = self._calc_table_attention(
                table_representation=table_representation,
                question_raw_embedding=question_raw_embedding,
                rnn_outputs=decoder_outputs,
                weights=table_attention_weights
            )
            return scores

    def _calc_ground_truth_embedding_index(self):
        """
        :return:
            [batch_size, max_question_length, 3]
        """
        with tf.name_scope("calc_ground_truth_embedding_index"):
            prefix_1 = tf.tile(
                tf.expand_dims(
                    tf.range(self._batch_size),
                    axis=1
                ),
                [1, self._max_question_length]
            )
            return tf.stack([prefix_1, self._ground_truth], axis=-1)

    def _calc_ground_truth_index(self):
        """
        Calc ground truth index
        :return:
            [batch_size, max_question_length, 3]
        """
        with tf.name_scope("calc_ground_truth_index"):
            prefix_1 = tf.tile(
                tf.expand_dims(
                    tf.range(self._batch_size),
                    axis=1
                ),
                [1, self._max_question_length]
            )
            prefix_2 = tf.tile(
                tf.expand_dims(
                    tf.range(self._max_question_length),
                    axis=0
                ),
                [self._batch_size, 1]
            )
            return tf.stack([prefix_1, prefix_2, self._ground_truth], axis=-1)

    def _test_predict_tag(self, table_representation, question_representation, question_raw_embedding, go_tag_embedding):
        """
        Predict tag in test phase
        :param table_representation:        [batch_size, table_size, table_extra_transform_dim]
        :param question_representation:     [batch_size, max_question_length, table_extra_transform_dim]
        :param question_raw_embedding:      [batch_size, max_question_length, combined_embedding_dim]
        :param go_tag_embedding:            [table_extra_transform_dim]
        :return:
            [batch_size, max_question_length]
        """
        with tf.name_scope("test_predict_tag"):

            decoder_cell = self._build_predict_rnn_decoder()
            table_attention_weights = self._initialize_table_attention_layer()
            # Shape: [batch_size, table_extra_transform_dim]
            expanded_go_tab_embedding = tf.tile(
                tf.expand_dims(go_tag_embedding, axis=0),
                multiples=[self._batch_size, 1]
            )

            with tf.name_scope("test_decode"):
                def __cond(_curr_ts, _prev_decoder_states, _tag_input, _prediction_score_array, _prediction_array):
                    return tf.less(_curr_ts, self._max_question_length)

                def __loop_body(_curr_ts, _prev_decoder_states, _tag_input, _prediction_score_array, _prediction_array):
                    """
                    :param _curr_ts:                Current time step
                    :param _prev_decoder_states:    Previous decoder states
                    :param _tag_input:              [batch_size, table_extra_transform_dim]
                    :param _prediction_array:       TensorArray
                    :return:
                    """
                    # Shape: [batch_size, 1, table_extra_transform_dim]
                    _question_input = tf.slice(
                        question_representation,
                        begin=[0, _curr_ts, 0],
                        size=[self._batch_size, 1, self._table_extra_transform_dim]
                    )
                    # Shape: [batch_size, 1, combined_embedding_dim]
                    _question_raw_embedding = tf.slice(
                        question_raw_embedding,
                        begin=[0, _curr_ts, 0],
                        size=[self._batch_size, 1, self._combined_embedding_dim]
                    )
                    # Shape: [batch_size, 1, table_extra_transform_dim*2]
                    _decoder_input = tf.concat(
                        values=[_question_input, _tag_input],
                        axis=-1
                    )
                    # _outputs: [batch_size, 1, predict_rnn_decoder_hidden_dim]
                    _outputs, _states = tf.nn.dynamic_rnn(
                        cell=decoder_cell,
                        inputs=_decoder_input,
                        initial_state=_prev_decoder_states,
                        dtype=tf.float32
                    )

                    _next_ts = tf.add(_curr_ts, 1)

                    # table_attention

                    # Shape: [batch_size, table_extra_transform_dim]
                    word_embedding_awared_rnn_outputs = tf.reshape(
                        tf.matmul(
                            tf.nn.relu(
                                tf.add(
                                    tf.matmul(
                                        tf.reshape(
                                            tf.concat(values=[_question_raw_embedding, _outputs], axis=-1),
                                            shape=[-1, self._combined_embedding_dim+self._question_rnn_encoder_hidden_dim]
                                        ),
                                        table_attention_weights["W"]
                                    ),
                                    table_attention_weights["b"]
                                )
                            ),
                            table_attention_weights["output_W"]
                        ),
                        shape=[self._batch_size, self._table_extra_transform_dim]
                    )

                    # Shape: [batch_size, table_size, table_extra_transform_dim]
                    expanded_word_embedding_awared_rnn_outputs = tf.reshape(
                        tf.tile(
                            tf.expand_dims(
                                word_embedding_awared_rnn_outputs,
                                axis=1
                            ),
                            multiples=[1, self._table_size, 1]
                        ),
                        shape=[self._batch_size, self._table_size,
                               self._table_extra_transform_dim]
                    )

                    # Shape: [batch_size, table_size]
                    _scores = tf.nn.softmax(
                        tf.reduce_sum(
                            tf.multiply(
                                expanded_word_embedding_awared_rnn_outputs,
                                table_representation
                            ),
                            axis=-1
                        ),
                        dim=-1
                    )

                    # Shape: [batch_size]
                    _prediction = tf.cast(tf.argmax(_scores, axis=-1), dtype=tf.int32)

                    # Shape: [batch_size]
                    _predicted_tag_score = tf.gather(_scores, _prediction)

                    _score_array = _prediction_score_array.write(_curr_ts, _predicted_tag_score)

                    _array = _prediction_array.write(_curr_ts, _prediction)

                    # Shape: [batch_size, 2]
                    _next_tag_index = tf.stack(
                        [
                            tf.reshape(tf.range(self._batch_size), [self._batch_size, 1]),
                            tf.reshape(_prediction, [self._batch_size, 1])
                        ],
                        axis=1
                    )

                    # Shape: [batch_size, table_extra_transform_dim]
                    _next_tag_inputs = tf.gather_nd(params=table_representation, indices=_next_tag_index)

                    return _next_ts, _states, _next_tag_inputs, _score_array, _array

                total_ts, final_states, last_tag_prediction, prediction_score_array, prediction_array = tf.while_loop(
                    body=__loop_body,
                    cond=__cond,
                    loop_vars=[
                        tf.constant(0, dtype=tf.int32),
                        decoder_cell.zero_state(self._batch_size),
                        expanded_go_tab_embedding,
                        tf.TensorArray(dtype=tf.float32, size=self._max_question_length),
                        tf.TensorArray(dtype=tf.int32, size=self._max_question_length)
                    ]
                )

                prediction_tensor = tf.transpose(
                    prediction_array.stack(name="prediction_tensor"),
                    perm=[1, 0]
                )
                prediction_score_tensor = tf.transpose(
                    prediction_score_array.stack(name="prediction_score_tensor"),
                    perm=[1, 0]
                )
                return prediction_tensor, prediction_score_tensor

    def _build_graph(self):
        self._build_input_nodes()
        self._set_dynamic_value()
        word_embedding, character_embedding, data_type_embedding, special_tag_embedding, go_tag_embedding = self._build_embedding()

        embedded_character = tf.nn.embedding_lookup(
            params=character_embedding,
            ids=self._word_character_matrix
        )
        # Shape: [None, character_embedding_dim]
        character_based_word_embedding = self._encode_word(
            embedded_character=embedded_character,
            word_length=self._word_character_length
        )

        combine_embedding_layer_params = self._initialize_combine_embedding_layer()

        # Shape: [batch_size, max_question_length, table_transform_embedding]
        # Shape: [batch_size, max_question_length, combined_embedding_dim]
        question_representation, question_raw_embedding = self._encode_question(
            character_based_word_embedding=character_based_word_embedding,
            word_embedding=word_embedding,
            combine_layer_params=combine_embedding_layer_params
        )

        # Shape: [batch_size, combined_embedding_dim]
        table_name_representation = self._encode_table_name(
            character_based_word_embedding=character_based_word_embedding,
            word_embedding=word_embedding,
            combine_layer_params=combine_embedding_layer_params
        )

        # Shape: [batch_size, max_column_name, combined_embedding_dim]
        column_name_representation = self._encode_column_name(
            character_based_word_embedding=character_based_word_embedding,
            word_embedding=word_embedding,
            combine_layer_params=combine_embedding_layer_params
        )

        # Shape: [batch_size, max_column_name, max_cell_value_num_per_col, cell_value_encoder_layer_2_dim]
        cell_value_representation = self._encode_cell_value(
            character_based_word_embedding=character_based_word_embedding,
            word_embedding=word_embedding,
            data_type_embedding=data_type_embedding,
            embedded_column_name=column_name_representation,
            combine_layer_params=combine_embedding_layer_params
        )

        # table:    [batch_size, table_extra_transform_dim]
        # column:   [batch_size, max_column_num, table_extra_transform_dim]
        table_name_representation, column_name_representation = self._transform_column_name_and_table_name(
            table_name_representation=table_name_representation,
            column_name_representation=column_name_representation
        )

        # Shape: [2, table_extra_dim]
        embedded_special_tag = self._transform_special_tag(special_tag_embedding)

        # Shape: [batch_size, table_size, table_extra_dim]
        table_representation = self._flatten_table(
            table_name_representation=table_name_representation,
            column_name_representation=column_name_representation,
            cell_value_representation=cell_value_representation,
            special_tag=embedded_special_tag
        )

        if self._is_test:
            self._predictions, self._scores = self._test_predict_tag(
                table_representation=table_representation,
                question_representation=question_representation,
                question_raw_embedding=question_raw_embedding,
                go_tag_embedding=go_tag_embedding
            )
            return

        # Shape: [batch_size, max_question_length, table_size]
        self._scores = self._train_predict_tag(
            table_representation=table_representation,
            question_representation=question_representation,
            question_raw_embedding=question_raw_embedding,
            go_tag_embedding=go_tag_embedding
        )

        # Predictions
        # Shape: [batch_size, max_question_length]
        self._predictions = tf.argmax(input=self._scores, axis=-1)

        # Loss
        # Calc Ground truth index
        # Shape: [batch_size, max_question_length, 3]
        ground_truth_index = self._calc_ground_truth_index()
        # Shape: [batch_size, max_question_length]
        probs = tf.reshape(
            tf.gather_nd(
                params=self._scores,
                indices=tf.reshape(
                    ground_truth_index,
                    shape=[-1, 3]
                )
            ),
            shape=[self._batch_size, self._max_question_length]
        )
        log_probs = tf.log(probs)
        self._loss = tf.negative(
            tf.reduce_mean(
                tf.reduce_sum(log_probs, axis=1)
            )
        )

        with tf.name_scope("back_propagation"):
            optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)

            # clipped at 5 to alleviate the exploding gradient problem
            self._gvs = optimizer.compute_gradients(self._loss)
            self._capped_gvs = [(tf.clip_by_value(grad, -self._gradient_clip, self._gradient_clip), var) for grad, var
                                in self._gvs]
            self._optimizer = optimizer.apply_gradients(self._capped_gvs)

    def _build_feed_dict(self, batch):
        feed_dict = dict()
        if not self._is_test:
            feed_dict[self._dropout_keep_prob] = 1 - self._dropout
            feed_dict[self._learning_rate] = batch.learning_rate
        else:
            feed_dict[self._dropout_keep_prob] = 1.
        feed_dict[self._questions_length] = batch.questions_length
        feed_dict[self._questions_char_ids] = batch.questions_char_ids
        feed_dict[self._questions_word_ids] = batch.questions_word_ids
        feed_dict[self._tables_name_word_ids] = batch.tables_name_word_ids
        feed_dict[self._tables_name_char_ids] = batch.tables_name_char_ids
        feed_dict[self._tables_name_length] = batch.tables_name_length
        feed_dict[self._column_name_length] = batch.column_name_length
        feed_dict[self._column_name_char_ids] = batch.column_name_char_ids
        feed_dict[self._column_name_word_ids] = batch.column_name_word_ids
        feed_dict[self._cell_value_length] = batch.cell_value_length
        feed_dict[self._cell_value_word_ids] = batch.column_word_ids
        feed_dict[self._cell_value_char_ids] = batch.column_char_ids
        feed_dict[self._ground_truth] = batch.ground_truth
        feed_dict[self._exact_match_matrix] = batch.exact_match_matrix
        feed_dict[self._column_data_type] = batch.column_data_type
        feed_dict[self._word_character_matrix] = batch.word_character_matrix
        feed_dict[self._word_character_length] = batch.word_character_length
        return feed_dict

    def train(self, batch):
        feed_dict = self._build_feed_dict(batch)
        return self._scores, self._predictions, self._loss, self._optimizer, feed_dict

    def predict(self, batch):
        feed_dict = self._build_feed_dict(batch)
        return self._predictions, feed_dict
