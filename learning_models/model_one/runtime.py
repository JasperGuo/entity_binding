# coding=utf8

import os
import time
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import Model
from data_iterator import DataIterator, VocabManager


def read_configuration(path):
    with open(path, "r") as f:
        return yaml.load(f)


def save_configuration(config, path):
    p = os.path.join(path, "config.yaml")
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


class ModelRuntime:

    def __init__(self, configuration):
        self._base_path = os.path.dirname(os.path.abspath(os.path.pardir))
        self._config = read_configuration(configuration)

        self._epoches = self._config["epoches"]
        self._batch_size = self._config["batch_size"]

        self._curr_time = str(int(time.time()))
        self._log_dir = os.path.abspath(self._config["log_dir"])
        self._result_log_base_pardir = os.path.abspath(self._config["result_log"])
        self._checkpoint_pardir = os.path.abspath(self._config["checkpoint_path"])

        for path in [self._log_dir, self._result_log_base_pardir, self._checkpoint_pardir]:
            if not os.path.exists(path):
                os.mkdir(path)

        self._result_log_base_path = os.path.join(self._result_log_base_pardir, self._curr_time)
        self._checkpoint_path = os.path.join(self._checkpoint_pardir, self._curr_time)

        self._checkpoint_file = os.path.join(self._checkpoint_path, "tf_checkpoint")
        self._best_checkpoint_file = os.path.join(self._checkpoint_path, "tf_best_checkpoint")

        os.mkdir(self._checkpoint_path)
        os.mkdir(self._result_log_base_path)
        save_configuration(self._config, self._result_log_base_path)
        save_configuration(self._config, self._checkpoint_path)

        self._test_data_iterator = None
        self._train_data_iterator = None
        self._dev_data_iterator = None

        self._word_vocab_manager = VocabManager(
            file=os.path.abspath(os.path.join(*([self._base_path] + self._config["vocab"]["word"])))
        )
        self._character_vocab_manager = VocabManager(
            file=os.path.abspath(os.path.join(*([self._base_path] + self._config["vocab"]["character"])))
        )
        self._data_type_vocab_manager = VocabManager(
            file=os.path.abspath(os.path.join(*([self._base_path] + self._config["vocab"]["data_type"])))
        )

        print("Word vocab size: ", self._word_vocab_manager.size)
        print("Char vocab size: ", self._character_vocab_manager.size)
        print("Data type vocab size: ", self._data_type_vocab_manager.size)

        self._pretrain_word_embedding = np.load(os.path.join(*([self._base_path] + self._config["embedding"]["word"])))
        self._session, self._train_model, self._test_model, self._saver, self._file_writer = None, None, None, None, None

    def init_session(self, checkpoint=None):
        self._session = tf.Session()

        with tf.variable_scope("entity_binding") as scope:
            self._train_model = Model(
                opts=self._config,
                word_vocab_size=self._word_vocab_manager.size,
                char_vocab_size=self._character_vocab_manager.size,
                column_data_type_num=self._data_type_vocab_manager.size,
                pretrain_word_embedding=self._pretrain_word_embedding,
                is_test=False
            )
            scope.reuse_variables()
            self._test_model = Model(
                opts=self._config,
                word_vocab_size=self._word_vocab_manager.size,
                char_vocab_size=self._character_vocab_manager.size,
                column_data_type_num=self._data_type_vocab_manager.size,
                pretrain_word_embedding=self._pretrain_word_embedding,
                is_test=True
            )
            self._saver = tf.train.Saver()
            if not checkpoint:
                init = tf.global_variables_initializer()
                self._session.run(init)
            else:
                self._saver.restore(self._session, checkpoint)
            self._file_writer = tf.summary.FileWriter(self._log_dir, self._session.graph)

    def _check_predictions(self, predictions, ground_truth):
        """
        :param predictions:     [batch_size, max_question_length]
        :param ground_truth:    [batch_size, max_question_length]
        :return:
        """
        p = np.array(predictions)
        g = np.array(ground_truth)
        result = np.sum(np.abs(p - g), axis=-1)
        correct = 0
        for r in result:
            if r == 0:
                correct += 1
        return correct

    def log(self, file, batch, predictions):
        with open(file, "a") as f:
            string = ""
            for t, p in zip(batch.ground_truth, predictions):
                string = "=======================\n"
                string += ("t: " + str(t) + "\n")
                string += ("p: " + str(p) + "\n")
            f.write(string)

    def train(self):
        self._batch = None
        try:
            best_operation_accuracy = 0
            last_updated_epoch = 0
            epoch_log_file = os.path.join(self._result_log_base_path, "epoch_result.log")
            file = os.path.join(self._result_log_base_path, "test_" + self._curr_time + ".log")
            curr_learning = self._config["learning_rate"]
            for epoch in tqdm(range(self._epoches)):
                self._train_data_iterator.shuffle()
                losses = list()
                total = 0
                train_correct = 0
                for i in tqdm(range(self._train_data_iterator.batch_per_epoch)):
                    batch = self._train_data_iterator.get_batch()
                    batch.learning_rate = curr_learning
                    self._batch = batch
                    # batch._print()
                    scores, predictions, loss, optimizer, feed_dict = self._train_model.train(batch)
                    scores, predictions, loss, optimizer = self._session.run(
                        (scores, predictions, loss, optimizer),
                        feed_dict=feed_dict
                    )
                    total += batch.size
                    train_correct += self._check_predictions(
                        predictions=predictions,
                        ground_truth=batch.ground_truth
                    )
                    losses.append(loss)
                    self.log(file=file, batch=batch, predictions=predictions)
                average_loss = np.average(np.array(losses))
                tqdm.write("epoch: %d, loss: %f, train_acc: %f" % (epoch, average_loss, train_correct/total))
        except (KeyboardInterrupt, SystemExit):
            # If the user press Ctrl+C...
            # Save the model
            # tqdm.write("===============================")
            # tqdm.write(str(self._batch.word_character_matrix))
            # tqdm.write("*******************************")
            # tqdm.write(str(self._batch.word_character_length))
            # tqdm.write("===============================")
            self._saver.save(self._session, self._checkpoint_file)
        except ValueError as e:
            print(e)
            self._batch._print()

    def run(self, is_test=False, is_log=False):
        if is_test:
            self._test_data_iterator = DataIterator(
                tables_file=os.path.abspath(os.path.join(*([self._base_path] + self._config["test"]["table"]))),
                questions_file=os.path.abspath(os.path.join(*([self._base_path] + self._config["test"]["question"]))),
                max_question_length=self._config["max_question_length"],
                max_word_length=self._config["max_word_length"],
                batch_size=self._config["batch_size"]
            )
        else:
            self._train_data_iterator = DataIterator(
                tables_file=os.path.abspath(os.path.join(*([self._base_path] + self._config["training"]["table"]))),
                questions_file=os.path.abspath(os.path.join(*([self._base_path] + self._config["training"]["question"]))),
                max_question_length=self._config["max_question_length"],
                max_word_length=self._config["max_word_length"],
                batch_size=self._config["batch_size"]
            )
            self.train()
            """
            self._dev_data_iterator = DataIterator(
                tables_file=os.path.abspath(os.path.join(*([self._base_path] + self._config["dev"]["table"]))),
                questions_file=os.path.abspath(os.path.join(*([self._base_path] + self._config["dev"]["question"]))),
                max_question_length=self._config["max_question_length"],
                max_word_length=self._config["max_word_length"],
                batch_size=self._config["batch_size"]
            )
            """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--conf", help="Configuration File")
    parser.add_argument("--checkpoint", help="Is Checkpoint ? Then checkpoint path ?", required=False)
    parser.add_argument("--test", help="Is test ?", dest="is_test", action="store_true")
    parser.add_argument("--no-test", help="Is test ?", dest="is_test", action="store_false")
    parser.set_defaults(is_test=False)
    parser.add_argument("--log", help="Is log ?", dest="is_log", action="store_true")
    parser.add_argument("--no-log", help="Is log ?", dest="is_log", action="store_false")
    parser.set_defaults(is_log=False)
    args = parser.parse_args()

    print(args.conf, args.checkpoint, args.is_test, args.is_log)

    runtime = ModelRuntime(args.conf)
    runtime.init_session(args.checkpoint)
    runtime.run(args.is_test, args.is_log)

