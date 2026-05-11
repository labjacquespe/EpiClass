"""EpiAtlas track-type and metadata constants.

Shared by the lazy fold factory and downstream consumers; lives here so
the eager epiatlas_treatment module can be retired without dragging
these along.
"""
from __future__ import annotations

import itertools

TRACKS_MAPPING = {
    "raw": ["pval", "fc"],
    "ctl_raw": [],
    "Unique_plusRaw": ["Unique_minusRaw"],
    "gembs_pos": ["gembs_neg"],
}

ACCEPTED_TRACKS = list(TRACKS_MAPPING.keys()) + list(
    itertools.chain.from_iterable(TRACKS_MAPPING.values())
)

LEADER_TRACKS = frozenset(["raw", "Unique_plusRaw", "gembs_pos"])
OTHER_TRACKS = frozenset(ACCEPTED_TRACKS) - LEADER_TRACKS

EPIRR_LABEL = "epirr_id"
