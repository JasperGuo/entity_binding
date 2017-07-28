# coding=utf8

import os
import time
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from model import Model
from data_iterator import VocabManager, Batch, BatchIterator

np.set_printoptions(threshold=np.nan)

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
        for idx, r in enumerate(result):
            if r == 0:
                correct += 1
                # tqdm.write(str(p[r]))
                # tqdm.write(str(g[r]))
                # tqdm.write("======================================")
        return correct

    def log(self, file, batch, predictions, is_detail=False):
        with open(file, "a") as f:
            string = ""
            for t, p, qid in zip(batch.ground_truth, predictions, batch.questions_ids):
                result = np.sum(np.abs(np.array(p) - np.array(t)), axis=-1)
                string += "=======================\n"
                string += ("id: " + str(qid) + "\n")
                string += ("t: " + str(t) + "\n")
                string += ("p: " + str(p) + "\n")
                string += ("Result: " + str(result == 0) + "\n")
                # string += ("s: " + str(scores) + "\n")
            f.write(string)

    def _epoch_log(self, file, num_epoch, train_accuracy, dev_accuracy, average_loss):
        """
        Log epoch
        :param file:
        :param num_epoch:
        :param train_accuracy:
        :param dev_accuracy:
        :param average_loss:
        :return:
        """
        with open(file, "a") as f:
            f.write("epoch: %d, train_accuracy: %f, dev_accuracy: %f, average_loss: %f\n" % (num_epoch, train_accuracy, dev_accuracy, average_loss))

    def test(self, data_iterator, is_log=False):
        tqdm.write("Testing...")
        total = 0
        correct = 0
        file = os.path.join(self._result_log_base_path, "test_" + self._curr_time + ".log")
        for i in tqdm(range(data_iterator.batch_per_epoch)):
            batch = data_iterator.get_batch()
            predictions, feed_dict = self._test_model.predict(batch)
            predictions = self._session.run(predictions, feed_dict=feed_dict)

            correct += self._check_predictions(
                predictions=predictions,
                ground_truth=batch.ground_truth
            )

            total += batch.size

            if is_log:
                self.log(
                    file=file,
                    batch=batch,
                    predictions=predictions
                )

        accuracy = float(correct)/float(total)
        tqdm.write("test_acc: %f" % accuracy)
        return accuracy

    def train(self):
        try:
            best_accuracy = 0
            epoch_log_file = os.path.join(self._result_log_base_path, "epoch_result.log")
            curr_learning = self._config["learning_rate"]
            for epoch in tqdm(range(self._epoches)):
                self._train_data_iterator.shuffle()
                losses = list()
                total = 0
                train_correct = 0
                file = os.path.join(self._result_log_base_path, "test_" + self._curr_time + "_" + str(epoch) + ".log")
                for i in tqdm(range(self._train_data_iterator.batch_per_epoch)):
                    batch = self._train_data_iterator.get_batch()
                    batch.learning_rate = curr_learning
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
                    """
                    if epoch % 10 == 0:
                        self.log(file=file, batch=batch, predictions=predictions)
                    """

                train_acc = train_correct / total

                self._dev_data_iterator.shuffle()
                dev_accuracy = self.test(self._dev_data_iterator, is_log=False)

                average_loss = np.average(np.array(losses))
                tqdm.write("epoch: %d, loss: %f, train_acc: %f, dev_acc: %f" % (epoch, average_loss, train_acc, dev_accuracy))

                if dev_accuracy > best_accuracy:
                    best_accuracy = dev_accuracy
                    self._saver.save(self._session, self._best_checkpoint_file)

                self._epoch_log(
                    file=epoch_log_file,
                    num_epoch=epoch,
                    train_accuracy=train_acc,
                    dev_accuracy=dev_accuracy,
                    average_loss=average_loss
                )

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

    def run(self, is_test=False, is_log=False):
        if is_test:
            self._test_data_iterator = BatchIterator(
                serialized_file=self._config["test"]["batches"]
            )
            self.test(self._test_data_iterator, True)
        else:
            self._train_data_iterator = BatchIterator(
                serialized_file=self._config["training"]["batches"]
            )
            self._dev_data_iterator = BatchIterator(
                serialized_file=self._config["dev"]["batches"]
            )
            self.train()


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

