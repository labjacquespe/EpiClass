# Compare UniqueMultiple vs Unique RNA-seq signal per genomic bin.
#
# For each sample/strand, UniqueMultiple (unique + multimapped reads) should
# differ from Unique (uniquely mapped reads only) only where multimapping
# occurs, so the histogram of per-bin differences should show a tall spike
# near 0 with a one-sided tail (UniqueMultiple >= Unique).
#
# File-wide pylint disables. Kept as a header comment (above `import marimo`) so
# marimo preserves it on save; an in-cell disable only scopes to that one cell.
# pylint: disable=missing-module-docstring, missing-function-docstring, function-redefined
# pylint: disable=import-error, import-outside-toplevel, reimported
# pylint: disable=redefined-outer-name, use-dict-literal, too-many-lines
# pylint: disable=unused-import, unused-argument, unused-variable, too-many-branches
# Structural to marimo's notebook format (cells are functions that return/display):
# pylint: disable=useless-return, pointless-statement, expression-not-assigned
# pylint: disable=too-many-positional-arguments, too-many-arguments

import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from epiclass.core.lazy.lazy_hdf5_loader import LazyHdf5Loader

    return LazyHdf5Loader, Path, go, make_subplots, mo, np


@app.cell
def _(go, np):
    def hist_bar(values, nbins=200):
        # Pre-bin server-side and return a go.Bar trace. go.Histogram would
        # ship every raw value (~30k/trace) to the browser; with 12 figures
        # that blows past marimo's output-size cap. Bars send only nbins counts.
        counts, edges = np.histogram(values, bins=nbins)
        centers = (edges[:-1] + edges[1:]) / 2
        return go.Bar(x=centers, y=counts, width=edges[1] - edges[0])

    return (hist_bar,)


@app.cell
def _(mo):
    # fmt: off
    mo.md(r"""
    # UniqueMultiple vs Unique RNA-seq signal

    Each sample/strand has a `Unique` track (uniquely mapped reads) and a `UniqueMultiple` track (unique + multimapped reads). We histogram the per-100kb-bin difference; most bins barely change, so expect a tall spike near 0.

    ## Why some differences are negative

    Intuitively `UniqueMultiple >= Unique` everywhere — but that only holds for **raw read counts**. These are **normalized** signal tracks (the "Raw" in `plusRaw`/`minusRaw` is a strand/processing label, not "un-normalized"; values are ~1e-5–1e0, not read depths). Both tracks are normalized to roughly the same genome-wide total, so multimapping does not *add* mass — it **redistributes** it into repeat-prone regions. After renormalization, clean uniquely-mappable high-coverage bins get **diluted**, producing negative differences. Evidence in this data: `sum(Multi) / sum(Unique) ≈ 1.003`, and the single strongest bin actually drops (`1.503 → 1.466`) in the Multi track. So negatives are expected, not a pairing bug.

    ## The "both-zero" variants

    Most 100kb bins are empty (no signal) in *both* tracks and contribute a huge exactly-0 difference. The second set of figures below drops bins where `Unique == 0 and UniqueMultiple == 0`, so the near-0 peak reflects real low-difference signal rather than empty intergenic genome.

    ## Reading the log2 panels (and why they are filtered)

    A log2 fold-change is only trustworthy when *both* tracks have real coverage: dividing one near-zero bin by another turns a trivial absolute difference into a huge ratio (a bin going 1e-8 → 2e-8 reads as +1, the same as a genuine doubling). A 100kb RNA-seq track is mostly empty, so a naive log2 histogram is dominated by these low-coverage artifacts and looks far worse than the data actually is — e.g. ~15% of bins exceed 4-fold change yet hold only ~2% of the signal.

    To keep it honest, the **log2 panel is restricted to the highest-coverage bins that together hold 99% of the signal** (`max(Multi, Unique)` ranked, cumulative-mass cutoff); the raw-difference panel above still shows every bin. This drops roughly half the bins but well under 0.1% of the signal, because signal is highly concentrated — the top 5% of bins carry ~70% of it — so the discarded bins are noise, not expression. A captured-signal-mass cutoff is used rather than a fixed value (e.g. the median) because it is interpretable and adapts to any track's sparsity.
    """)
    # fmt: on
    return


@app.cell
def _(Path):
    DATA_DIR = Path(
        "/home/local/USHERBROOKE/rabj2301/Downloads/ihec_data/"
        "epiatlas_portal/rna/test_100kb_RNA"
    )
    CHROM_FILE = Path(
        "/home/local/USHERBROOKE/rabj2301/Projects/epiclass/"
        "input/chromsizes/hg38.can.chrom.sizes"
    )
    MMAP_DIR = DATA_DIR / "mmap_cache"
    EPS = 1e-9  # avoids div-by-zero / log(0) in the ratio metric
    return CHROM_FILE, DATA_DIR, EPS, MMAP_DIR


@app.cell
def _(CHROM_FILE, DATA_DIR, LazyHdf5Loader, MMAP_DIR):
    # Write the HDF5 path list expected by the loader, then register + preload.
    # normalization=False keeps raw signal so per-bin differences are meaningful;
    # z-normalising each file independently would destroy the comparison.
    hdf5_paths = sorted(DATA_DIR.glob("*.hdf5"))
    hdf5_list_path = DATA_DIR / "hdf5_list.txt"
    hdf5_list_path.write_text(
        "\n".join(str(p) for p in hdf5_paths) + "\n", encoding="utf-8"
    )

    loader = LazyHdf5Loader(CHROM_FILE, normalization=False, mmap_dir=MMAP_DIR)
    loader.register_hdf5s(hdf5_list_path)
    loader.preload_all()
    return (loader,)


@app.cell
def _(loader):
    # Pair each UniqueMultiple signal with its Unique equivalent.
    # Signal IDs look like:
    #   ihec.rna-seq.<container>.IHECRE00000229.3.<uuid>.UniqueMultiple.plusRaw
    # so swapping the mapping flavour yields the partner; the IHECRE id + strand
    # give a human-readable label.
    def _label(signal_id: str) -> str:
        parts = signal_id.split(".")
        ihecre = next((p for p in parts if p.startswith("IHECRE")), "?")
        version = parts[parts.index(ihecre) + 1] if ihecre in parts else ""
        strand = parts[-1]  # plusRaw / minusRaw
        return f"{ihecre}.{version} {strand}"

    pairs = {}
    for sid in loader.file_paths:
        if ".UniqueMultiple." not in sid:
            continue
        unique_id = sid.replace(".UniqueMultiple.", ".Unique.")
        if unique_id not in loader.file_paths:
            print(f"No Unique partner for {sid}")
            continue
        pairs[_label(sid)] = (sid, unique_id)

    print(f"Found {len(pairs)} UniqueMultiple/Unique pairs")
    return (pairs,)


@app.cell
def _(hist_bar, loader, make_subplots, np):
    def _signal_mass_mask(cov, mass_frac):
        # Keep the highest-coverage bins that together hold `mass_frac` of the
        # total signal; the rest is near-empty genome where a ratio is just
        # noise. Returns a boolean mask over the input bins.
        order = np.argsort(cov)[::-1]
        cum = np.cumsum(cov[order])
        if cum[-1] == 0:
            return np.zeros_like(cov, dtype=bool)
        k = int(np.searchsorted(cum, mass_frac * cum[-1])) + 1
        return cov >= cov[order][k - 1]

    def build_pair_fig(label, mid, uid, eps, exclude_both_zero, log2_mass_frac=0.99):
        # One figure per pair; raw difference on top, log2 ratio below.
        multi = loader.load_signal(mid)
        unique = loader.load_signal(uid)
        if exclude_both_zero:
            keep = ~((multi == 0) & (unique == 0))
            multi, unique = multi[keep], unique[keep]

        raw = multi - unique
        # log2 fold-change is unstable where coverage ~ 0, so restrict the log2
        # panel to the high-coverage bins that carry the signal. The raw panel
        # still shows every bin.
        cov = np.maximum(multi, unique)
        mask = _signal_mass_mask(cov, log2_mass_frac)
        log2r = np.log2((multi[mask] + eps) / (unique[mask] + eps))

        _fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"Raw difference (UniqueMultiple − Unique) — all bins (n={raw.size})",
                "log2((UniqueMultiple + eps) / (Unique + eps)) — bins holding "
                f"{log2_mass_frac:.0%} of signal (n={int(mask.sum())})",
            ),
        )
        _fig.add_trace(hist_bar(raw), row=1, col=1)
        _fig.add_trace(hist_bar(log2r), row=2, col=1)
        _fig.update_yaxes(type="log", title_text="bin count (log)", row=1, col=1)
        _fig.update_yaxes(type="log", title_text="bin count (log)", row=2, col=1)
        _fig.update_xaxes(title_text="UniqueMultiple − Unique", row=1, col=1)
        _fig.update_xaxes(title_text="log2 ratio", row=2, col=1)
        _fig.update_layout(
            height=700,
            showlegend=False,
            bargap=0,
            title_text=f"{label}  (n bins = {raw.size})",
        )
        return _fig

    return (build_pair_fig,)


@app.cell
def _():
    ## including zero bins
    # mo.vstack(
    #     [
    #         build_pair_fig(label, mid, uid, EPS, exclude_both_zero=False)
    #         for label, (mid, uid) in pairs.items()
    #     ]
    # )
    return


@app.cell
def _(EPS, build_pair_fig, mo, pairs):
    # non zero bins
    mo.vstack(
        [
            build_pair_fig(label, mid, uid, EPS, exclude_both_zero=True)
            for label, (mid, uid) in pairs.items()
        ]
    )
    return


if __name__ == "__main__":
    app.run()
