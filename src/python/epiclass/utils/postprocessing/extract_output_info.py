"""
Extract specific information from what epiML prints.

Useful to get several results for the Excel sheet at once.
"""
import argparse
import re
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("epiML_output", type=Path, help="File to extract info from.")
    return arg_parser.parse_args()


class InvalidTokenError(Exception):
    """Raised when the token is not valid."""


class EpiMLOutputReader:
    """Read epiML output files and extract their information."""

    SIZE_TOKENS = frozenset(["training", "validation", "test"])
    EXAMPLES_TOKEN = "Examples"
    TRAINING_TOKEN = "epoch"
    METRICS_TOKENS = frozenset(["Training", "Validation", "Test"])
    HYPERPARAMS_TOKENS = frozenset(
        [
            "Nb",
            "Layers",
            "batch_size:",
            "early_stop_limit:",
            "is_training:",
            "keep_prob:",
            "l1_scale:",
            "l2_scale:",
            "learning_rate:",
            "measure_frequency:",
            "training_epochs:",
        ]
    )
    METRICS = frozenset(["Accuracy", "Precision", "Recall", "f1_score", "MCC"])

    def __init__(self):
        self._file = None
        self._info = {}
        self._hyperparams_fields = self._init_hyperparams_fields()
        self._tokens = self._init_tokens()
        self._current_line = ""

    def _init_tokens(self):
        """Return set of all tokens"""
        return set([self.EXAMPLES_TOKEN, self.TRAINING_TOKEN]).union(
            self.SIZE_TOKENS, self.METRICS_TOKENS, self.HYPERPARAMS_TOKENS
        )

    def _init_hyperparams_fields(self):
        """Return token:field_name dict for hyperparameters."""
        fields = {token: token.strip(":") for token in self.HYPERPARAMS_TOKENS}
        fields.update({"Nb": "nb_layers", "Layers": "layers_size"})
        return fields

    def print_info(self, fields=None):
        """Print info in order of given fields. Print all info with keys if no field is given."""
        if fields is None:
            for key, val in sorted(self._info.items()):
                print(f"{key} : {val}")
        else:
            infos = [self._info.get(field, "--") for field in fields]
            print("\t".join(infos))

    def read_file(self, file):
        """Read file and extract important information."""
        self._file = open(file, "r", encoding="utf-8")
        self._info = {}  # empty if another file was read before

        while True:
            try:
                self._next_line()
            except StopIteration:
                self._current_line = ""
                break

            first_word = self._get_current_first_word()
            if first_word in self._tokens:
                self._read_section(first_word)

        self._file.close()

    def _get_current_first_word(self):
        """Return string before first split on a space."""
        return self._current_line.rstrip("\n").split(" ", 1)[0]

    def _next_line(self):
        """Advance to next file line."""
        self._current_line = next(self._file)

    def _read_section(self, token):
        """Choose section reading method."""
        if token in self.HYPERPARAMS_TOKENS:
            self._read_hyperparams()
        elif token in self.SIZE_TOKENS:
            self._read_set_size()
        elif token in self.METRICS_TOKENS:
            self._read_metrics()
        elif token == self.TRAINING_TOKEN:
            self._read_training()
        elif token == self.EXAMPLES_TOKEN:
            self._read_examples()
        else:
            raise InvalidTokenError(f"Invalid token: {token}")

    def _read_hyperparams(self):
        """Extract hyperparameters from multiple lines"""
        while True:
            first_word = self._get_current_first_word()
            if first_word in self._hyperparams_fields:
                field_name = self._hyperparams_fields[first_word]
                field_info = self._current_line.strip("\n").split(" ")[-1]
                self._info[field_name] = field_info

                self._next_line()
            else:
                break

    def _read_set_size(self):
        """Extract set size from "[SetName] size [SetSize]" line."""
        dataset, word, size = self._current_line.strip("\n").split(" ")
        if word == "size":
            self._info[f"{dataset}_size"] = size
        else:
            raise InvalidTokenError(
                f"Not a set size section. Problematic token:{dataset}"
            )

    def _read_examples(self):
        """Extract the total number of examples from "For a total of [Nb] examples" line."""
        while True:
            self._next_line()
            if self._get_current_first_word() == "For":
                self._info["nb_examples"] = self._current_line.strip("\n").split(" ")[4]
                break

    def _read_training(self):
        """Extract last epoch number and date from
        "epoch [Nb], batch training accuracy [float], validation accuracy [float] [timestamp]"
        lines. Also extract training time if present just after.
        """
        self._info["date"] = self._current_line.split(" ")[-2]

        last_epoch = ""
        while True:
            self._next_line()
            first_word, epoch, _ = self._current_line.split(" ", 2)
            if first_word == "epoch":
                last_epoch = epoch
            else:
                self._info["last_epoch"] = last_epoch.strip(",")
                break

        if self._is_training_time_line():
            self._read_training_time()
        else:
            print("Training time not present after training. Continuing")

    def _is_training_time_line(self):
        """Return boolean based on if the line gives the training time."""
        first_word, second_word = self._current_line.split(" ")[0:2]
        return first_word == "training" and second_word == "time:"

    def _read_training_time(self):
        """Extract training time from "training time: [timedelta]" line."""
        match = re.search(r"(\w{1,2}):(\w{1,2}):(\w{2}).", self._current_line)
        if match:
            self._info["training_time"] = "{}h{}m{}s".format(*match.groups())

    def _read_metrics(self):
        """Extract metrics from multiple lines."""
        dataset = self._get_current_first_word()
        while True:
            self._next_line()
            line = self._current_line.strip("\n").split(":")
            first_word = line[0]

            if first_word in self.METRICS:
                field_name = f"{dataset}_{first_word}".lower()
                self._info[field_name] = line[1].strip(" ")

            # another metrics section
            elif self._get_current_first_word() in self.METRICS_TOKENS:
                self._read_metrics()
            else:
                break


def main():
    """Miaw"""

    options = parse_arguments()

    wanted_infos = [
        "resolution",
        "nb_layers",
        "layers_size",
        "assembly",
        "nb_examples",
        "nb_classes",
        "training_size",
        "batch_size",
        "learning_rate",
        "l2_scale",
        "last_epoch",
        "training_time",
        "training_accuracy",
        "validation_accuracy",
        "validation_precision",
        "validation_recall",
        "validation_f1_score",
        "validation_mcc",
        "date",
    ]

    reader = EpiMLOutputReader()
    reader.read_file(options.epiML_output)

    reader.print_info(wanted_infos)


if __name__ == "__main__":
    main()
