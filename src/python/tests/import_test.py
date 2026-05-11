"""Test imports for all python files."""
# pylint: disable=unused-import, import-outside-toplevel, consider-using-from-import, reimported
import pytest


@pytest.mark.slow
def test_imports() -> None:
    r"""Test that all core modules and main imports work together.

    Gathered using 'find -type f -name "*.py" -exec cat {} + | grep 'import ' | awk '{$1=$1;print}' | sort -u | grep -vE "^from \.|__future__|^\#"|\($'.
    """
    import abc
    import argparse
    import collections
    import concurrent.futures
    import copy
    import csv
    import datetime
    import decimal
    import glob
    import importlib.resources as resources
    import io
    import itertools
    import json
    import logging
    import marshal
    import math
    import multiprocessing
    import os
    import os.path
    import pickle
    import random
    import re
    import shutil
    import subprocess
    import sys
    import tarfile
    import tempfile
    import traceback
    import typing
    import warnings
    from collections import Counter, defaultdict
    from collections.abc import ItemsView, KeysView, ValuesView
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
    from datetime import datetime
    from difflib import SequenceMatcher as SM
    from functools import partial, wraps
    from importlib import metadata
    from inspect import signature
    from pathlib import Path
    from typing import (
        IO,
        Any,
        Callable,
        Collection,
        Dict,
        FrozenSet,
        Generator,
        Iterable,
        List,
        Optional,
        Sequence,
        Set,
        Tuple,
        TypeVar,
        Union,
    )

    import comet_ml  # needed first because special snowflake
    import h5py
    import matplotlib
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.typing as npt
    import pandas as pd
    import paramiko
    import plotly.express as px
    import plotly.graph_objects as go
    import pyBigWig
    import pytorch_lightning as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
    import pytorch_lightning.callbacks as pl_callbacks
    import scipy
    import shap
    import sklearn
    import sklearn.metrics
    import skops.io as skio
    import statsmodels.api as sm
    import torch
    import torch.nn.functional as F
    import torchmetrics
    import umap
    from git import Repo
    from imblearn.over_sampling import RandomOverSampler
    from IPython.display import display
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    from numpy.typing import ArrayLike
    from pytorch_lightning import loggers as pl_loggers
    from scipy import signal
    from scipy.special import softmax  # type: ignore
    from scipy.stats.mstats import winsorize
    from scp import SCPClient
    from sklearn import preprocessing
    from sklearn.decomposition import IncrementalPCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        make_scorer,
        matthews_corrcoef,
        roc_auc_score,
    )
    from sklearn.model_selection import (
        StratifiedGroupKFold,
        StratifiedKFold,
        cross_val_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        LabelBinarizer,
        LabelEncoder,
        OneHotEncoder,
        StandardScaler,
    )
    from sklearn.svm import SVC, LinearSVC
    from statsmodels.multivariate.manova import MANOVA
    from tabulate import tabulate
    from torch import Tensor, nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchinfo import summary
    from umap.umap_ import nearest_neighbors

    from epiclass.argparseutils.DefaultHelpParser import (
        DefaultHelpParser as ArgumentParser,
    )
    from epiclass.argparseutils.directorychecker import DirectoryChecker
    from epiclass.core import analysis, metadata
    from epiclass.core.analysis import write_pred_table
    from epiclass.core.confusion_matrix import ConfusionMatrixWriter
    from epiclass.core.data.dataset import DataSet
    from epiclass.core.data.eager import KnownData, UnknownData
    from epiclass.core.data_source import HDF5_RESOLUTION, EpiDataSource
    from epiclass.core.epiatlas_treatment import (
        ACCEPTED_TRACKS,
        TRACKS_MAPPING,
        EpiAtlasFoldFactory,
    )
    from epiclass.core.loaders.dataset_factory import DataSetFactory
    from epiclass.core.loaders.hdf5_loader import Hdf5Loader
    from epiclass.core.metadata import Metadata, UUIDMetadata
    from epiclass.core.model_pytorch import LightningDenseClassifier
    from epiclass.core.shap_values import NN_SHAP_Handler
    from epiclass.core.trainer import MyTrainer, define_callbacks
    from epiclass.core.types import SomeData, TensorData
    from epiclass.utils import modify_metadata
    from epiclass.utils.bed_utils import bins_to_bed_ranges, write_to_bed
    from epiclass.utils.check_dir import create_dirs
    from epiclass.utils.general_utility import (
        get_valid_filename,
        read_paths,
        write_hdf5_paths_to_file,
        write_md5s_to_file,
    )
    from epiclass.utils.metadata_utils import (
        DP_ASSAYS,
        EPIATLAS_ASSAYS as epiatlas_assays,
        EPIATLAS_CATS,
        count_combinations,
        count_pairs,
    )
    from epiclass.utils.my_logging import log_dset_composition, log_pre_training
    from epiclass.utils.postprocessing.augment_predict_file import add_coherence
    from epiclass.utils.shap.shap_analysis import feature_overlap_stats
    from epiclass.utils.shap.shap_utils import extract_shap_values_and_info
    from epiclass.utils.time import time_now, time_now_str
    from epiclass.utils.torch_data import create_torch_datasets
