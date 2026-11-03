"""EpiAtlas constants testing module."""
from __future__ import annotations

import importlib

from epiclass.core import epiatlas_constants


def test_epirr_label_from_env(monkeypatch):
    """$EPIRR_LABEL overrides the default field; unset or empty gives the default."""
    try:
        monkeypatch.setenv("EPIRR_LABEL", "EpiRR")
        assert importlib.reload(epiatlas_constants).EPIRR_LABEL == "EpiRR"

        for value in [None, ""]:
            if value is None:
                monkeypatch.delenv("EPIRR_LABEL")
            else:
                monkeypatch.setenv("EPIRR_LABEL", value)
            assert importlib.reload(epiatlas_constants).EPIRR_LABEL == "epirr_id"
    finally:
        monkeypatch.undo()
        importlib.reload(epiatlas_constants)
