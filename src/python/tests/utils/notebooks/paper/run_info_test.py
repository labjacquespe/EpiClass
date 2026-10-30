"""Tests for the paper's training-run helpers (utils/notebooks/paper/run_info.py).

run_info relies on conventions set elsewhere: the folder layout of k-fold runs, and the
lines the training code prints to the SLURM logs. The tests check both sides:

- the conventions as the code that creates them produces them (log_pre_training,
  MyTrainer.save_model_path, the model repr), so a change there fails here;
- the real paper runs, through ``fixtures/run_info/`` (the real run tree, reduced to the
  log lines run_info reads, with the values run_info gave on the real tree; see
  ``tests/fixtures_gen/make_run_info_fixture.py``), so a change in run_info or in the
  folder naming shows which runs are affected.

Note: collect_exp_keys excludes log files whose *absolute* path matches "shap|predict",
so the test names here (which pytest uses for tmp_path) avoid those words.
"""
# pylint: disable=missing-function-docstring
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock

import pandas as pd
import pytest

from epiclass.core.model_pytorch import LightningDenseClassifier
from epiclass.core.prediction_files import experiment_id_from_checkpoint
from epiclass.core.trainer import MyTrainer
from epiclass.utils import my_logging
from epiclass.utils.notebooks.paper.paper_utilities import (
    ASSAY,
    CELL_TYPE,
    SplitResultsHandler,
)
from epiclass.utils.notebooks.paper.run_info import (
    collect_exp_keys,
    extract_experiment_info,
    extract_experiment_keys_from_output_files,
    extract_input_sizes_from_output_files,
    get_split_exp_keys,
    parse_run_dir,
    task_name_from_run_dir,
)
from tests.epilap_test_data import FIXTURES_DIR
from tests.fixtures_gen.make_run_info_fixture import (
    exp_keys_table,
    feature_set_dirs,
    log_values_tables,
)

RUN_INFO_FIXTURE = FIXTURES_DIR / "run_info"
RUN_TREE = RUN_INFO_FIXTURE / "dfreeze_v2"

KEY_A = "76e601be43a640238f78d9f37d142587"
KEY_B = "4c624cfdec2b4e169a1afc73df15a0cb"


def _key(i: int) -> str:
    """Return a fake 32-hex comet-ml experiment key."""
    return f"{i:032x}"


def _checkpoint_line(split_name: str, exp_key: str, quoted: bool = False) -> str:
    """Checkpoint path line of a SLURM .o log (2024 runs: unquoted; current: quoted)."""
    path = f"/logs/release/fs/task_1l_3000n/10fold-oversampling/{split_name}/EpiLaP/{exp_key}/checkpoints/epoch=1-step=2.ckpt"
    return f"Saving model to '{path}'" if quoted else f"Saving model to {path}"


def _write_log(run_dir: Path, lines: List[str], name: str = "output_job1_run.o") -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / name
    log_file.write_text("\n".join(lines) + "\n", encoding="utf8")
    return log_file


def _read_expected(name: str) -> pd.DataFrame:
    return pd.read_csv(
        RUN_INFO_FIXTURE / f"expected_{name}.tsv",
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


class TestRunDirNames:
    """Run folder (relative to its feature set folder) -> task name."""

    @pytest.mark.parametrize(
        "relpath, category, name_parts, task_name",
        [
            ("assay_epiclass_1l_3000n/10fold-oversampling", "assay_epiclass", [], "assay_epiclass"),
            ("assay_epiclass_1l_3000n/10fold", "assay_epiclass", [], "assay_epiclass"),
            ("assay_epiclass_1l_3000n/11c/10fold-oversampling", "assay_epiclass", ["11c"], "assay_epiclass_11c"),
            ("assay_epiclass_1l_3000n/7c/10fold", "assay_epiclass", ["7c"], "assay_epiclass_7c"),
            ("assay_epiclass_1l_3000n/chip-seq-only/10fold-oversampling", "assay_epiclass", ["chip-seq-only"], "assay_epiclass_chip-seq-only"),
            ("harmonized_donor_sex_1l_3000n/w-mixed/10fold-oversampling", "harmonized_donor_sex", ["w-mixed"], "harmonized_donor_sex_w-mixed"),
            # Only the exact k-fold folder names are dropped
            ("harmonized_sample_ontology_intermediate_1l_3000n/10fold-oversampling-unique_assay", "harmonized_sample_ontology_intermediate", ["10fold-oversampling-unique_assay"], "harmonized_sample_ontology_intermediate_10fold-oversampling-unique_assay"),
            # Other classifiers (no "_1l_3000n")
            ("assay_epiclass/predict-10fold", "assay_epiclass", ["predict-10fold"], "assay_epiclass_predict-10fold"),
        ],
    )  # fmt: skip
    def test_real_layouts(self, relpath, category, name_parts, task_name):
        assert parse_run_dir(Path(relpath)) == (category, name_parts)
        assert task_name_from_run_dir(Path(relpath)) == task_name

    @pytest.mark.parametrize(
        "relpath",
        [
            "assay_epiclass_1l_3000n/7c/10fold-2",
            "harmonized_sample_ontology_intermediate_1l_3000n/10fold-oversampling-unique_assay/h3k27me3_only",
        ],
    )
    def test_more_than_one_name_part_raises(self, relpath):
        _, name_parts = parse_run_dir(Path(relpath))  # parsing alone does not raise
        assert len(name_parts) == 2
        with pytest.raises(ValueError, match="Too many parts"):
            task_name_from_run_dir(Path(relpath))

    def test_k_sets_the_dropped_folder_names(self):
        relpath = Path("assay_epiclass_1l_3000n/5fold-oversampling")
        assert task_name_from_run_dir(relpath, k=5) == "assay_epiclass"
        assert task_name_from_run_dir(relpath) == "assay_epiclass_5fold-oversampling"

    def test_names_used_by_the_paper_notebooks(self):
        # fig1 renames f"{ASSAY}_11c" and reads CELL_TYPE results under these names.
        assay_11c = Path(f"{ASSAY}_1l_3000n/11c/10fold-oversampling")
        cell_type = Path(f"{CELL_TYPE}_1l_3000n/10fold-oversampling")
        assert task_name_from_run_dir(assay_11c) == f"{ASSAY}_11c"
        assert task_name_from_run_dir(cell_type) == CELL_TYPE


class TestSplitResultsTaskNames:
    """SplitResultsHandler.general_split_metrics names and filters runs with run_info."""

    @staticmethod
    def _write_run(results_dir: Path, run_dir: str, n_splits: int = 2) -> None:
        for i in range(n_splits):
            split_dir = results_dir / run_dir / f"split{i}"
            split_dir.mkdir(parents=True)
            pd.DataFrame(
                {"True class": ["a", "b"], "Predicted class": ["a", "a"], "a": [0.9, 0.6], "b": [0.1, 0.4]},
                index=[_key(2 * i), _key(2 * i + 1)],
            ).to_csv(split_dir / "validation_prediction.csv")  # fmt: skip

    @pytest.fixture(name="results_dir")
    def results_dir_fixture(self, tmp_path) -> Path:
        results_dir = tmp_path / "hg38_100kb_all_none"
        self._write_run(results_dir, f"{ASSAY}_1l_3000n/11c/10fold-oversampling")
        self._write_run(results_dir, f"{ASSAY}_1l_3000n/7c/10fold-oversampling")
        self._write_run(results_dir, f"{CELL_TYPE}_1l_3000n/10fold-oversampling")
        self._write_run(results_dir, f"{CELL_TYPE}_1l_3000n/10fold")  # not oversampled
        # Malformed: two name parts
        self._write_run(
            results_dir, "harmonized_donor_sex_1l_3000n/a/b/10fold-oversampling"
        )
        return results_dir

    def _task_names(self, results_dir: Path, **kwargs) -> List[str]:
        split_results = SplitResultsHandler().general_split_metrics(
            results_dir=results_dir,
            merge_assays=False,
            return_type="split_results",
            **kwargs,
        )
        return sorted(split_results)  # type: ignore

    def test_task_names_and_filters(self, results_dir):
        include = {"include_categories": [ASSAY, CELL_TYPE]}
        assert self._task_names(results_dir, **include) == sorted(
            [f"{ASSAY}_11c", f"{ASSAY}_7c", CELL_TYPE]
        )
        assert self._task_names(results_dir, exclude_names=["7c"], **include) == sorted(
            [f"{ASSAY}_11c", CELL_TYPE]
        )

    def test_malformed_run_only_raises_when_not_filtered_out(self, results_dir):
        # The category filters run before the name is validated.
        self._task_names(results_dir, exclude_categories=["harmonized_donor_sex"])
        with pytest.raises(ValueError, match="Too many parts"):
            self._task_names(results_dir)


class TestExperimentInfo:
    """(split, experiment key) from the checkpoint path lines of the .o logs."""

    def test_unquoted_line(self):
        assert extract_experiment_info(_checkpoint_line("split3", KEY_A)) == (
            "split3",
            KEY_A,
        )

    def test_line_printed_by_trainer(self, tmp_path, capsys):
        ckpt = (
            tmp_path
            / "split3"
            / "EpiClass"
            / KEY_A
            / "checkpoints"
            / "epoch=1-step=2.ckpt"
        )
        trainer = SimpleNamespace(
            checkpoint_callback=SimpleNamespace(
                best_model_path=str(ckpt), last_model_path=""
            ),
            best_checkpoint_file=tmp_path / "best_checkpoint.list",
        )
        MyTrainer.save_model_path(trainer)  # type: ignore
        line = capsys.readouterr().out.strip()
        assert line.startswith("Saving model to")
        assert extract_experiment_info(line) == ("split3", KEY_A)

    def test_agrees_with_prediction_files(self):
        # Same checkpoint layout as experiment_id_from_checkpoint (prediction CSV tags).
        ckpt = f"/run/split0/EpiClass/{KEY_B}/checkpoints/last.ckpt"
        _, exp_key = extract_experiment_info(f"Saving model to {ckpt}")  # type: ignore
        assert exp_key == experiment_id_from_checkpoint(ckpt) == KEY_B

    def test_no_validation_run(self):
        line = f"Saving model to /logs/task/complete_no_valid_oversample/EpiLaP/{KEY_A}/checkpoints/last.ckpt"
        assert extract_experiment_info(line) == ("complete_no_valid_oversample", KEY_A)

    @pytest.mark.parametrize(
        "line",
        [
            "The current experiment key is " + KEY_A,
            "Saving model to EpiLaP/checkpoints/last.ckpt",
        ],
    )
    def test_other_lines_give_none(self, line):
        assert extract_experiment_info(line) is None


class TestLogValueExtractors:
    """Values read from the .o logs, as the training code prints them."""

    def test_experiment_key_printed_by_log_pre_training(
        self, tmp_path, capsys, monkeypatch
    ):
        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.setattr(
            my_logging.subprocess, "check_output", lambda *_, **__: "v1\n"
        )
        logger = MagicMock()
        logger.experiment.get_key.return_value = KEY_A
        to_log = {"category": ASSAY, "hdf5_resolution": "100000", "loading_time": "1"}
        my_logging.log_pre_training(logger=logger, to_log=to_log, step=None)

        _write_log(tmp_path / "fs" / "run", capsys.readouterr().out.splitlines())
        assert extract_experiment_keys_from_output_files(tmp_path) == {"fs": {KEY_A}}

    def test_input_size_from_model_repr(self, tmp_path):
        model = LightningDenseClassifier(
            input_size=37, output_size=2, mapping={0: "a", 1: "b"}, hparams={}, hl_units=8
        )
        _write_log(tmp_path / "fs" / "run", str(model).splitlines())
        assert extract_input_sizes_from_output_files(tmp_path) == {"fs": {37}}

    def test_two_input_sizes_in_one_log_raises(self, tmp_path):
        lines = [
            "    (1): Linear(in_features=37, out_features=8, bias=True)",
            "    (1): Linear(in_features=38, out_features=8, bias=True)",
        ]
        _write_log(tmp_path / "fs" / "run", lines)
        with pytest.raises(ValueError, match="more than one unique line"):
            extract_input_sizes_from_output_files(tmp_path)


class TestSplitExperimentKeys:
    """{split: experiment key} of a task's k-fold oversampled run."""

    K = 3

    def _write_run_logs(
        self, fs_dir: Path, run_dir: str, keys: Dict[str, List[str]]
    ) -> None:
        lines = [
            _checkpoint_line(split_name, exp_key)
            for split_name, split_keys in keys.items()
            for exp_key in split_keys
        ]
        _write_log(fs_dir / run_dir, lines)

    def _keys(self, offset: int) -> Dict[str, List[str]]:
        return {f"split{i}": [_key(offset + i)] for i in range(self.K)}

    def test_selects_the_task_run(self, tmp_path):
        fs_dir = tmp_path / "hg38_100kb_all_none"
        self._write_run_logs(
            fs_dir, f"{ASSAY}_1l_3000n/11c/{self.K}fold-oversampling", self._keys(0)
        )
        self._write_run_logs(fs_dir, f"{ASSAY}_1l_3000n/11c/{self.K}fold", self._keys(10))
        self._write_run_logs(
            fs_dir, f"{ASSAY}_1l_3000n/7c/{self.K}fold-oversampling", self._keys(20)
        )

        split_keys = get_split_exp_keys(fs_dir, f"{ASSAY}_11c", k=self.K)
        assert split_keys == {f"split{i}": _key(i) for i in range(self.K)}

        split_keys = get_split_exp_keys(
            fs_dir, f"{ASSAY}_11c", k=self.K, oversampling=False
        )
        assert split_keys == {f"split{i}": _key(10 + i) for i in range(self.K)}

    def test_missing_split_raises(self, tmp_path):
        fs_dir = tmp_path / "fs"
        keys = self._keys(0)
        del keys["split2"]
        self._write_run_logs(fs_dir, f"{ASSAY}_1l_3000n/{self.K}fold-oversampling", keys)
        with pytest.raises(ValueError, match="not one key per split"):
            get_split_exp_keys(fs_dir, ASSAY, k=self.K)

    def test_rerun_split_raises(self, tmp_path):
        fs_dir = tmp_path / "fs"
        keys = self._keys(0)
        keys["split1"].append(_key(99))
        self._write_run_logs(fs_dir, f"{ASSAY}_1l_3000n/{self.K}fold-oversampling", keys)
        with pytest.raises(ValueError, match="not one key per split"):
            get_split_exp_keys(fs_dir, ASSAY, k=self.K)

    def test_unknown_task_raises(self, tmp_path):
        fs_dir = tmp_path / "fs"
        self._write_run_logs(
            fs_dir, f"{ASSAY}_1l_3000n/{self.K}fold-oversampling", self._keys(0)
        )
        with pytest.raises(ValueError, match="not one key per split"):
            get_split_exp_keys(fs_dir, CELL_TYPE, k=self.K)

    def test_excluded_log_paths(self, tmp_path):
        fs_dir = tmp_path / "fs"
        self._write_run_logs(
            fs_dir, f"{ASSAY}_1l_3000n/{self.K}fold-oversampling", self._keys(0)
        )
        _write_log(
            fs_dir / f"{ASSAY}_1l_3000n" / "shap" / "run",
            [_checkpoint_line("split0", _key(50))],
        )
        keys = {
            exp_key for keys in collect_exp_keys(fs_dir).values() for _, exp_key in keys
        }
        assert keys == {_key(i) for i in range(self.K)}


class TestPaperRuns:
    """run_info on the paper's real run tree (fixtures/run_info/)."""

    # Feature sets of Ext Fig 2 (fig1 notebook: feature_sets_14), under mixed/
    EXT_FIG2_SETS = [
        "hg38_10mb_all_none_1mb_coord",
        "hg38_100kb_random_n316_none",
        "hg38_1mb_all_none",
        "hg38_100kb_random_n3044_none",
        "hg38_100kb_all_none",
        "hg38_gene_regions_100kb_coord_n19864",
        "hg38_10kb_random_n30321_none",
        "hg38_regulReg_allCorr_n30k",
        "hg38_1kb_random_n30321_none",
        "hg38_cpg_topvar_200bp_n30k_wrong_coordinates",
        "hg38_10kb_all_none",
        "hg38_regulReg_allCorr_n303k",
        "hg38_1kb_random_n303114_none",
        "hg38_cpg_topvar_200bp_n303k_wrong_coordinates",
    ]

    def test_task_names_of_all_run_dirs(self):
        expected = _read_expected("task_names")
        for row in expected.itertuples():
            relpath = Path(row.run_dir)
            name_parts = row.name_parts.split(";") if row.name_parts else []
            assert parse_run_dir(relpath) == (row.category, name_parts), row.run_dir
            try:
                task_name = task_name_from_run_dir(relpath)
            except ValueError:
                task_name = "ValueError"
            assert task_name == row.task_name, f"{row.feature_set_dir}/{row.run_dir}"

    def test_experiment_keys(self):
        expected = _read_expected("exp_keys")
        assert exp_keys_table(RUN_TREE).astype(str).equals(expected)

    def test_input_sizes_and_node_jobs(self):
        for name, table in log_values_tables(RUN_TREE).items():
            assert table.astype(str).equals(_read_expected(name)), name

    def test_feature_set_dirs(self):
        expected_dirs = set(_read_expected("exp_keys")["feature_set_dir"])
        fixture_dirs = {
            fs.relative_to(RUN_TREE).as_posix() for fs in feature_set_dirs(RUN_TREE)
        }
        assert expected_dirs <= fixture_dirs

    @pytest.mark.parametrize("category", [ASSAY, CELL_TYPE])
    def test_ext_fig2_runs_have_one_key_per_split(self, category):
        expected = _read_expected("exp_keys")
        all_keys = set()
        for feature_set in self.EXT_FIG2_SETS:
            task_name = category
            if category == ASSAY and feature_set == "hg38_100kb_all_none":
                task_name = f"{ASSAY}_11c"
            split_keys = get_split_exp_keys(RUN_TREE / "mixed" / feature_set, task_name)

            assert sorted(split_keys) == [f"split{i}" for i in range(10)]
            rows = expected[expected["feature_set_dir"] == f"mixed/{feature_set}"]
            for split_name, exp_key in split_keys.items():
                assert exp_key in set(rows.loc[rows["split"] == split_name, "exp_key"])
            all_keys.update(split_keys.values())
        assert len(all_keys) == len(self.EXT_FIG2_SETS) * 10

    def test_symlinked_task_dirs(self):
        # hg38_100kb_all_none/<task> links to mixed/: os.walk (general_split_metrics)
        # follows the link, Path.rglob (extract_* readers) does not.
        link = RUN_TREE / "hg38_100kb_all_none" / f"{ASSAY}_1l_3000n"
        assert link.is_symlink()
        assert (
            link.resolve()
            == (
                RUN_TREE / "mixed" / "hg38_100kb_all_none" / f"{ASSAY}_1l_3000n"
            ).resolve()
        )
