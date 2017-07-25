# coding=utf8

import os
import time
import yaml
import argparse
from data_iterator import DataIterator


def read_configuration(path):
    with open(path, "r") as f:
        return yaml.load(f)


def save_configuration(config, path):
    p = os.path.join(path, "config.yaml")
    with open(p, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


class ModelRuntime:

    def __init__(self, configuration):

        self._base_path = os.path.pardir
        self._config = read_configuration(configuration)

        self._epoches = self._config["epoches"]
        self._batch_size = self._config["batch_size"]

        self._curr_time = str(int(time.time()))
        self._log_dir = os.path.abspath(self._config["log_dir"])
        self._result_log_base_pardir = os.path.abspath(os.path.join(os.path.curdir, self._config["result_log"]))
        self._checkpoint_pardir = os.path.abspath(os.path.join(os.path.curdir, self._config["checkpoint_path"]))

        for path in [self._log_dir, self._result_log_base_pardir, self._checkpoint_pardir]:
            if not os.path.exists(path):
                os.mkdir(path)

        self._result_log_base_path = os.path.join(self._result_log_base_pardir, self._curr_time)
        self._checkpoint_path = os.path.join(self._checkpoint_pardir, self._curr_time)
        self._checkpoint_file = os.path.join(os.path.curdir, self._checkpoint_path, "tf_checkpoint")
        self._best_checkpoint_file = os.path.join(os.path.curdir, self._checkpoint_path, "tf_best_checkpoint")

        os.mkdir(self._checkpoint_path)

        os.mkdir(self._result_log_base_path)
        save_configuration(self._config, self._result_log_base_path)
        save_configuration(self._config, self._checkpoint_path)

        self._test_data_iterator = None
        self._train_data_iterator = None
        self._dev_data_iterator = None

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
    # runtime.init_session(args.checkpoint)
    runtime.run(args.is_test, args.is_log)

