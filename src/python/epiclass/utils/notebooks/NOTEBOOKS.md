# Notebook Summary

All notebooks live under `src/python/epiclass/utils/notebooks/`. Jupyter (`.ipynb`) notebooks are grouped by directory below; interactive marimo (`.py`) apps are in the **Marimo apps** section at the end.

---

## Comet-ML (`comet/`)

Notebooks for interacting with the Comet-ML experiment tracking API and summarizing the metadata it stores.

### `comet/collect_all.ipynb`

Downloads all Comet-ML experiments for the project, flattens nested JSON metrics/hyperparameters into a flat DataFrame, and fixes oversampling label inconsistencies. Produces a full JSON archive and filtered CSVs per classifier type.

- **Input:** Comet-ML REST API (authenticated)
- **Output:** `comet_ml_all_experiments_full_YYYY-MM-DD.json`, filtered CSVs

### `comet/collect_comet_summary.ipynb`

Recovery notebook: parses SLURM `.e` error/log files to reconstruct training metrics for experiments where Comet-ML upload failed. Extracts accuracy, F1, and other metrics from raw log text.

- **Input:** SLURM `.e` log files from HPC
- **Output:** Recovered metrics CSV

### `comet/collect_input_size.ipynb`

Retrieves the neural network input size hyperparameter from Comet-ML experiment records to track how input dimensionality varied across training configurations.

- **Input:** Comet-ML API
- **Output:** `input_sizes.csv`

### `comet/collect_specific_exp.ipynb`

Selectively collects metadata, metrics, and hyperparameters for a specific subset of Comet-ML experiments provided via a CSV list. Used for targeted re-analysis of particular runs.

- **Input:** Comet-ML API, CSV of target experiment keys
- **Output:** Filtered metrics/metadata CSV

### `comet/group_training_times.ipynb`

Extracts and summarizes neural network training wall-clock times from Comet-ML experiment JSON archives. Computes mean ± std training time per classifier configuration for supplementary table ST17.

- **Input:** `comet_ml_all_experiments_full_YYYY-MM-DD.json.xz`, supplementary table CSV
- **Output:** Training time summary table

---

## Preprocessing (`preprocessing/`)

Notebooks that produce or transform data files consumed by downstream analyses.

### `preprocessing/CNV_treatment.ipynb`

Creates CNV signature BED files from TCGA binary bedgraph data (produced by `preprocessing/segment_to_binary_bedgraph.ipynb`). Computes z-scores of SHAP region overlap with cancer CNV hotspots against random BED controls.

- **Input:** TCGA binary bedgraph files, SHAP BED files, chromosome sizes
- **Output:** CNV signature BED files, z-score enrichment figures

### `preprocessing/regions_lengths.ipynb`

Analyzes and plots the distribution of feature (genomic region) lengths across various SHAP feature sets to characterize the typical genomic span of features identified as important by the classifier.

- **Input:** SHAP BED files per class/assay
- **Output:** Feature length distribution plots

### `preprocessing/segment_to_binary_bedgraph.ipynb`

Transforms TCGA copy number variation (CNV) segment files into binary bedgraph format. Fills genomic coverage gaps using bedtools genomecov to produce complete genome-wide binary CNV tracks.

- **Input:** TCGA CNV segment files (TSV), reference genome chromosome sizes
- **Output:** Binary bedgraph files per sample

### `preprocessing/TF_functions.ipynb`

Acquires transcription factor functional annotations (activator, repressor, dual, etc.) from curated TF databases to annotate SHAP-selected TF binding sites with their regulatory function.

- **Input:** TF function databases (downloaded programmatically), list of TF names from SHAP analysis
- **Output:** TF function annotation TSV

---

## Metadata (`metadata/`)

Notebooks for assembling, merging, comparing, or analyzing sample metadata. Also covers data-split validation (stratification) since that's about how the metadata is partitioned.

### `metadata/analyze_metadata.ipynb`

Analyzes metadata composition of the EpiATLAS training set: distributions and counts across assay, cell type, sex, life stage, biomaterial type, and disease categories. The complementary creation/augmentation steps were split out into `create_metadata.ipynb`.

- **Input:** dfreeze JSON metadata
- **Output:** Composition tables and printouts

### `metadata/c-a_metadata.ipynb`

Handles ChIP-Atlas public database metadata: creates a consensus `core7` database label column, downloads additional GEO metadata, and computes biosample type/tissue composition for ChIP-Atlas predictions.

- **Input:** ChIP-Atlas metadata TSV, GEO API, EpiATLAS metadata JSON
- **Output:** Augmented ChIP-Atlas metadata CSV, composition plots

### `metadata/create_metadata.ipynb`

Builds new metadata JSON files (split off from `analyze_metadata.ipynb` to preserve the creation steps): (1) creates a metadata JSON for imputed pval ChIP-seq files by joining EpiRR-level metadata with imputed file MD5 lists; (2) sanity-checks that imputed vs observed pval datasets have similar cell-type composition; (3) merges new Martin/PE cell type labels into the v2 metadata to produce a `_newCT.json` variant; (4) merges pre-purge classifier predictions with the official BadQual mislabel list and pivots per-track predictions wide for review.

- **Input:** dfreeze metadata JSONs, imputed MD5 list, Martin cell type TSV, BadQual CSV, pre-purge 10fold prediction CSV
- **Output:** `hg38_epiatlas_imputed_pval_chip_2024-02.json`, `*_newCT.json`, `official_BadQual_augmented.csv`

### `metadata/encode_metadata_creation.ipynb`

Recreates ENCODE metadata from scratch using the ENCODE REST API. Downloads file, experiment, biosample, and biosample_type metadata for all ENCODE ChIP, RNA, and WGBS files; merges them with prefixed column names; removes revoked entries; adds `assay_epiclass` and `in_epiatlas` columns.

- **Input:** EBI EpiRR API, ENCODE portal REST API, old ChIP metadata CSV
- **Output:** `encode_full_metadata_2025-02_no_revoked.csv/.json`, hg38 freeze1 subset

### `metadata/evaluate_biospecimen_similarity.ipynb`

Uses ontology-based semantic similarity (CL, UBERON, EFO/CLO/NTR via OLS4 API) to compare biospecimen sources between EpiATLAS and ENCODE samples. Builds a unified NXOntology graph; computes pairwise Lin and Jaccard similarity. Classifies biosample overlap into tiers: exact, high (>0.7), moderate (>0.4), low (>0.2), distinct, unmapped.

- **Input:** ENCODE metadata freeze1 XZ, EpiATLAS training metadata JSON, OBO ontology files
- **Output:** `biospecimen_comparison_pairwise.csv`, `biospecimen_comparison_best_matches.csv`, `overlap_summary_weighted.csv`

### `metadata/evaluate_stratification.ipynb`

Validates the 10-fold StratifiedGroupKFold cross-validation setup. Checks for EpiRR/UUID data leakage between folds and verifies all classes appear in each fold. Identified and documented a scikit-learn v1.5.2 bug (missing classes in some folds), confirmed fixed in v1.8.0.

- **Input:** EpiATLAS metadata JSON, fold assignment files
- **Output:** Fold composition tables, leakage check reports

### `metadata/import_data.ipynb`

Downloads neural network training result files from a remote HPC cluster (narval mount) via Python file copying. Also compares training metadata with the official IHEC EpiATLAS metadata to identify discrepancies, missing UUIDs, and extra UUIDs across metadata versions.

- **Input:** HPC narval filesystem (mounted), official IHEC EpiATLAS metadata CSVs from GitHub
- **Output:** Discrepancy CSVs per category, UUID diff lists

### `metadata/metadata_bias_analysis.ipynb`

Quantifies metadata bias: tests whether classification labels can be predicted from other metadata features alone (assay, project, biomaterial type) using LR, RandomForest, and SVM with OneHotEncoding and 10-fold CV. Also computes third-factor correlations: per-(assay, cell type) metadata distribution vs global, correlated with cell type accuracy vectors.

- **Input:** EpiATLAS metadata JSON, merged prediction CSV
- **Output:** `metadata_bias_analysis_results.csv`, per-assay third-factor Pearson correlation CSVs

### `metadata/metadata_diff.ipynb`

Detailed comparison of EpiClass training metadata against official IHEC EpiATLAS metadata versions (v1.0–v2.0). Identifies changed labels (version-to-version mislabels), resolved unknowns, and unresolved unknowns for sex and life stage categories. Finds UUIDs present in official EpiATLAS metadata but absent from training, and vice versa.

- **Input:** Training metadata JSON, official IHEC metadata CSVs from GitHub
- **Output:** Diff CSVs per category/version/status, UUID diff lists, `training_metadata_vs_official.json`

---

## Signal Analysis (`signal/`)

Notebooks for analyzing raw epigenomic signal values in HDF5 files, including dimensionality reduction projections and signal-based QC.

### `signal/analyze_corr.ipynb`

Explores pairwise signal correlation patterns between histone marks (H3K27ac vs H3K4me3). Produces violin plots of correlation distributions across assay groups.

- **Input:** Precomputed correlation matrix XLSX (`avr_median_ca_epiatlas_2023_05_31--corr100kb.xlsx`)
- **Output:** Violin plots

### `signal/analyze_hdf5_vals.ipynb`

Multi-purpose notebook: pairwise Pearson correlations between HDF5 signal files, global bin-level metrics, SHAP feature genomic position analysis, and BED-to-gene intersections for supplementary tables.

- **Input:** NPZ signal files, HDF5 file lists, metadata JSON, ChromScore NPZ, GFF3 intersect TSVs
- **Output:** Correlation histograms, violin plots, merged TSV tables

### `signal/analyze_predictions.ipynb`

Creates pivot tables by sex, assay, and cell type; builds confusion matrices; plots chrY signal intensity vs predicted sex class to validate sex classifier output.

- **Input:** Sex-classifier prediction CSV, chrY signal CSVs
- **Output:** Pivot CSVs, violin/subplot figures

### `signal/chrY_analysis.ipynb`

Benchmarks the MLP sex classifier against naive classifiers (median-threshold, GMM, logistic regression) using chrY and chrX z-scores. Documents that GMM fails due to heavy-tailed data; logistic regression is recommended as a baseline.

- **Input:** recount3 metadata TSV, mean chrY signal values CSV
- **Output:** Classifier comparison figures, performance tables

### `signal/chrY_zscore.ipynb`

Computes chrY z-scores normalized by assay and track type distribution. Creates confusion-matrix-style violin plots and KDE density plots comparing predicted vs true sex labels.

- **Input:** EpiATLAS metadata JSON, chrY/chrX signal CSV, sex prediction CSV
- **Output:** Z-score CSVs, violin subplot figures, KDE density plots

### `signal/epiatlas_qc_analysis.ipynb`

Analyzes WGBS QC metrics for EpiATLAS samples that cluster anomalously with input controls in 3D UMAP space. Compares per-metric distributions (BS conversion rate, etc.) between suspicious clusters and all other WGBS samples using violin plots.

- **Input:** `EpiATLAS_wgbs_qc_summary.csv`, UMAP cluster MD5 files, EpiATLAS metadata JSON
- **Output:** Per-metric violin plots per cluster vs background

### `signal/pca_plot.ipynb`

Generates PCA projection plots for EpiATLAS and ChIP-Atlas datasets, including comparisons between core and non-core assay samples to visualize epigenomic signal separability in principal component space.

- **Input:** HDF5 signal files or precomputed PCA NPZ, metadata JSON
- **Output:** PCA scatter plots (HTML/PNG)

### `signal/umap_plot.ipynb`

Generates UMAP embedding plots for EpiATLAS and ChIP-Atlas datasets. Compares core vs non-core assay samples in UMAP space with flexible metadata coloring.

- **Input:** HDF5 signal files or precomputed UMAP coordinates, metadata JSON
- **Output:** UMAP scatter plots (HTML/PNG)

---

## Model Interpretation (`interpretability/`)

Notebooks for inspecting model-internal signals: first-layer weight distributions and SHAP feature importance, plus downstream biological enrichment of SHAP-selected regions.

### `interpretability/analyze_weights.ipynb`

Examines the distribution of first-layer neural network weights. Produces violin plots of absolute weight magnitudes to characterize what the model learned at the input layer.

- **Input:** Saved PyTorch Lightning model checkpoints
- **Output:** `weights_dist.png`, `weights_description.csv`

### `interpretability/shap/analyze_shaps.ipynb`

Full single-fold SHAP pipeline: extracts SHAP value matrices, selects top-N most important genomic features, computes feature overlap statistics across classes, and writes BED files for downstream enrichment analysis.

- **Input:** SHAP NPZ/HDF5 files, EpiATLAS metadata JSON, chromosome sizes file
- **Output:** BED files per class, overlap statistics CSVs, feature rank CSVs, UpSet plots

### `interpretability/shap/analyze_shaps_over_folds.ipynb`

Aggregates SHAP feature importance across all 10 cross-validation folds. Computes per-fold and cross-fold intersection matrices, produces heatmaps, and selects globally stable features present across folds.

- **Input:** Per-fold SHAP BED files, chromosome sizes file
- **Output:** Intersection matrix CSVs, heatmap PNGs, globally-stable BED files

### `interpretability/shap/analyze_shaps_subsets.ipynb`

Identifies which top SHAP features are unique to the ChIP raw track type versus other track types (fold-change, p-value). Computes per-subset feature counts and uniqueness.

- **Input:** `feature_count.json` files per track subset
- **Output:** `unique_features_count_union.csv`, `feature_count_per_subset.csv`

### `interpretability/shap/cancer_regions_overlap.ipynb`

Tests whether SHAP-selected genomic regions are enriched for cancer-associated genes. Computes z-scores of overlap with the COSMIC cancer gene list against 200 randomly sampled BED controls. Also intersects SHAP regions with TCGA CNV signatures.

- **Input:** SHAP BED files, COSMIC cancer gene annotations, TCGA CNV BED files, GFF3 annotations
- **Output:** Z-score enrichment figures, overlap statistics CSVs

### `interpretability/shap/chromscore.ipynb`

Analyzes ChromScore (chromatin activity metrics derived from bigwig files) for SHAP-selected genomic regions vs background. Uses Welch's t-test, Brunner-Munzel test, and Cohen's d effect size to quantify chromatin activity enrichment.

- **Input:** ChromScore NPZ file, SHAP BED files, chromosome sizes
- **Output:** ChromScore comparison figures, statistical test results CSVs

### `interpretability/shap/go_enrichment_shap_vs_naive_baseline.ipynb`

Compares GO enrichment quality from SHAP-selected genomic regions against a naive differential region selection baseline. The naive baseline uses per-biospecimen and per-(assay, biospecimen) one-vs-rest effect sizes (mean difference / std). Runs bedtools intersect and gProfiler GO enrichment for all approaches and summarizes results with -log10 p-value statistics.

- **Input:** EpiATLAS 100kb signals as NPZ, SHAP BED files from tar.gz archive, GFF3 annotations, gProfiler API
- **Output:** BED files per biospecimen/assay-biospecimen pair, GO enrichment TSVs, `GO_enrichment_comparison_extended.tsv`

### `interpretability/shap/prep_shap_runs.ipynb`

Prepares HDF5 file lists for SHAP computation jobs on the HPC cluster: generates background sample lists (used as SHAP reference distribution) and evaluation sample lists.

- **Input:** EpiATLAS metadata JSON, HDF5 file directories
- **Output:** `shap_background_*.list`, `shap_eval_*.list` files per class/assay

### `interpretability/shap/profile_bed.ipynb`

GO enrichment pipeline for BED files: filters GFF3 gene annotations, intersects BED features with gene bodies via bedtools, then submits gene lists to gProfiler REST API for GO/pathway enrichment.

- **Input:** BED feature files, GFF3 gene annotation, gProfiler API
- **Output:** Gene lists per BED set, GO enrichment TSVs, bedtools GFF-intersect TSVs

### `interpretability/shap/sample_ontology_shap_ranks_graph.ipynb`

Graphs SHAP feature ranks per cell type, focusing on the top 3 features by median rank across samples. Includes ChromHMM repressive feature analysis to characterize the chromatin state context of top SHAP features.

- **Input:** SHAP rank TSVs, ChromHMM annotation files, cell type labels
- **Output:** SHAP rank figures per cell type, ChromHMM feature characterization plots

### `interpretability/shap/shap_analysis.ipynb`

Detailed SHAP analysis covering: comparison of SHAP background distribution vs training metadata distribution, feature rank graphing, and ChromHMM-based analysis of repressive chromatin features among top SHAP features.

- **Input:** SHAP NPZ/HDF5 files, ChromHMM annotation BED files, EpiATLAS metadata JSON
- **Output:** Background vs training metadata comparison figures, SHAP rank plots, ChromHMM enrichment figures

---

## Predictions (`predictions/`)

Notebooks that assemble or analyze classifier prediction outputs — across folds, classifier algorithms, hyperparameter sweeps, and external public databases.

### `predictions/c-a_pred_analysis.ipynb`

Analyzes classifier predictions on the ChIP-Atlas public database. Loads the ChIP-Atlas prediction CSV, computes database consensus agreement across GEO/Cistrome/NGS/C-A sources, creates UpSet plots of database overlap, uses classifier predictions to resolve "no consensus" labels, and computes mislabel breakdowns by GEO study (GSE). Produces per-assay accuracy/F1 metrics across multiple tasks (assay 7c, sex, cancer, biomat, life stage).

- **Input:** `ChIP-Atlas_predictions_*_merge_metadata_freeze1.csv.xz`, ChIP-Atlas metadata
- **Output:** Metrics TSVs, confusion matrices, UpSet plots, `gse_count_incorrect_pred_*.tsv`

### `predictions/compare_alt_tracks_results.ipynb`

Compares classifier accuracy when training with different ChIP-seq track type subsets (all tracks, no fold-change, p-value only, raw signal only). Accuracy remains near 100% regardless of subset.

- **Input:** Comet-ML experiment results per track subset
- **Output:** Accuracy comparison figures

### `predictions/confidence_threshold.ipynb`

Analyzes how prediction confidence score thresholds affect dataset composition and classifier accuracy for EpiATLAS (cross-validation) and public databases (ENCODE, ChIP-Atlas, recount3). Produces Supplementary Figures 1H, 1I, and 6.

- **Input:** Merged prediction CSVs for EpiATLAS and public DBs, metadata JSONs
- **Output:** Threshold vs accuracy/retention plots (HTML/PNG/SVG)

### `predictions/correlation_analysis.ipynb`

Analyzes the relationship between pairwise Pearson signal correlations and EpiClass mislabel predictions for ChIP-Atlas and ENCODE samples. For each mislabeled sample, computes mean correlation to EpiATLAS assay groups to test whether mislabels correlate more with the predicted class than the annotated class.

- **Input:** `mislabels_C-A&ENCODE_assay7.csv`, correlation matrix `.mat` file, EpiATLAS metadata JSON
- **Output:** Per-sample and per-mislabeled-assay violin plots (HTML/PNG/SVG)

### `predictions/encode_chip_QC.ipynb`

Reviewer-response analysis validating that EpiClass low-confidence predictions (high Input-class probability) correlate with conventional ENCODE ChIP-seq quality metrics. Runs per-target Spearman correlations with 9,999-permutation p-values between Input-class prediction probability and FRiP, NSC, and peak count QC metrics.

- **Input:** ENCODE experiment quality metrics JSON, complete ENCODE predictions CSV
- **Output:** Two-panel boxplot (core histone marks vs non-core TFs), `per_target_qc_correlations_*.csv`

### `predictions/encode_pred_analysis.ipynb`

Comprehensive ENCODE prediction analysis across 7 classification tasks (assay 7c/11c/13c, cell type, sex, life stage, cancer, biomaterial type). Computes structured accuracy/F1 tables per assay, stacked bar charts of non-core assay category composition, and confusion matrices. Maps non-core assays to functional categories (trx_reg, heterochrom, polycomb, splicing, insulator, other/mixed).

- **Input:** Prediction CSVs per task (ChIP/RNA/WGBS), ENCODE metadata, non-core assay category CSV
- **Output:** Metrics TSVs, stacked bar charts, confusion matrices, `complete_encode_predictions_augmented_2025-02_metadata.csv.gz`

### `predictions/merge_dfreeze_results.ipynb`

Merges all per-fold training prediction result files from the HPC cluster into a single unified DataFrame augmented with chrY z-scores and metadata.

- **Input:** Per-fold prediction CSVs from HPC, chrY z-score CSV
- **Output:** `merged_pred_results_all_2.1_chrY_zscores.csv`

### `predictions/merge_prediction_results.ipynb`

Merges ENCODE-specific augmented prediction files (across ChIP, RNA, WGBS assay types) into a single DataFrame for ENCODE-wide analysis.

- **Input:** Per-assay ENCODE prediction augmented CSVs
- **Output:** `encode_predictions_merged_results_V2.csv`

### `predictions/non_core_pred.ipynb`

Analyzes predictions from the 9c-nc (9 core + non-core) assay classifier within EpiATLAS. Checks what fraction of non-core files are predicted as core assays and what fraction of core files are mislabeled as non-core. For non-core mislabels, checks whether the second-highest prediction matches the correct class.

- **Input:** 9c-nc prediction CSV, non-core assay category labels
- **Output:** Prediction summary tables, composition figures

### `predictions/pred_cell_types.ipynb`

Analyzes mispredictions from various cell type metadata groupings. Examines which cell type grouping schemes lead to more or fewer classification errors and characterizes the nature of those errors.

- **Input:** Cell type prediction CSVs, metadata JSON
- **Output:** Misprediction analysis figures, cell type grouping comparison tables

### `predictions/recount3_pred_analysis.ipynb`

Analyzes classifier predictions on the recount3 RNA-seq public database. Computes per-assay accuracy and F1 metrics, biosample type composition of predictions, and results across multiple classification tasks.

- **Input:** recount3 prediction CSVs per classification task, recount3 metadata
- **Output:** Metrics TSVs, biosample composition figures, confusion matrices

### `predictions/regularization_data.ipynb`

Analyzes the effect of L1 regularization strength and dropout rate on classifier performance. Generates figures showing the regularization-accuracy trade-off used to select final hyperparameters.

- **Input:** Comet-ML experiment results for regularization sweep runs
- **Output:** L1/dropout effect figures

### `predictions/result_exploration.ipynb`

Exploratory analysis of correct vs incorrect predictions: per-assay, per-track-type, per-cell-type, and per-sex error rates to characterize where and why the classifier makes mistakes.

- **Input:** Merged prediction CSV with metadata
- **Output:** Error rate tables and figures per metadata category

---

## Paper Figures — Development (`paper/figures-dev/`)

Draft figure notebooks superseded by `paper/paper-final/` versions. Kept because they contain analyses not reproduced in the finals — see note under each.

### `paper/figures-dev/flagship_figures.ipynb`

Development version of the flagship figures. Covers cell type accuracy per assay (violin + boxplot, colored per cell type), SHAP input bin signal analysis, and cell-type-specific per-assay training results (unique-assay 100kb cross-validation).

**Note:** This content is not reproduced in `paper-final/flagship_figures.ipynb`, which covers different panels (reduced feature sets, GO enrichment, ChromScore, public DB inference).

### `paper/figures-dev/paper_fig1.ipynb`

Development version of Figure 1. Covers prediction score violin plots per assay, multi-algorithm performance boxplots (NN/LR/LGBM/LinearSVC/RF), per-assay accuracy boxplots, and imputed vs observed training accuracy scatter graphs for ChIP-Atlas.

**Note:** The imputation analysis section (observed vs imputed training, confidence threshold vs samples-conserved curves) is not reproduced in `paper-final/fig1.ipynb`.

### `paper/figures-dev/paper_fig2.ipynb`

Development version of Figure 2. Covers sex and life stage mislabel analyses, chrY z-score figures, reduced feature set performance metrics, and track type comparison figures.

**Note:** The reduced feature set and track type comparison content may not be reproduced in `paper-final/fig2.ipynb`.

### `paper/figures-dev/rebuttal.ipynb`

Analyses conducted in response to peer reviewer comments. Includes comparison of UUID-based vs EpiRR-based cross-validation fold assignment strategies, and additional reviewer-requested analyses.

- **Input:** EpiATLAS metadata JSON, fold assignment files, prediction CSVs
- **Output:** Rebuttal figures (HTML/PNG/SVG)

---

## Paper Figures — Final (`paper/paper-final/`)

Production notebooks used to generate the published paper figures.

### `paper/paper-final/fig1.ipynb`

Final production Figure 1. Includes ROC curves, biospecimen-level metrics, reduced feature set scatter graphs, and multi-algorithm performance comparisons.

- **Input:** 10-fold prediction result files, metadata JSON, feature set result CSVs
- **Output:** Final Figure 1 panels (HTML/PNG/SVG)

### `paper/paper-final/fig2.ipynb`

Final production Figure 2. Covers MLP performance across classification tasks, donor sex label correction with chrY z-scores, GP-Age life stage predictions, GO enrichment heatmap, ChromScore analysis, and CNV signature overlap.

- **Input:** Age prediction results, GO enrichment TSVs, CNV overlap CSVs, SHAP intersection matrices, chrY signal CSV
- **Output:** Final Figure 2 panels (HTML/PNG/SVG)

### `paper/paper-final/fig3.ipynb`

Final production Figure 3. Covers ENCODE non-core assay prediction analysis (including insulator assay), pie charts of public DB prediction composition, and public database prediction summary figures.

- **Input:** ENCODE/ChIP-Atlas/recount3 prediction CSVs, non-core assay category CSV
- **Output:** Final Figure 3 panels (HTML/PNG/SVG)

### `paper/paper-final/flagship_figures.ipynb`

Complete flagship paper figure production notebook. Generates all figures required for the EpiATLAS flagship publication: cell type accuracy per assay for different training region sets (CpG, regulatory, gene, 100kb), GO enrichment heatmap, ChromScore violin per biospecimen, and public DB inference bar chart.

- **Input:** 10kb and 100kb training results, SHAP NPZ files, ChromScore NPZ, metadata JSON
- **Output:** All flagship paper figures (SVG/PNG/HTML)

### `paper/paper-final/flagship_figures_simple.ipynb`

Simplified version of `flagship_figures.ipynb` with reduced code complexity, producing a streamlined subset of the flagship paper figures. Intended as a cleaner reference version.

- **Input:** Same as `flagship_figures.ipynb`
- **Output:** Simplified flagship paper figures (SVG/PNG/HTML)

### `paper/paper-final/generate_all_predictions_files.ipynb`

Creates supplementary prediction files for the paper. Merges prediction results from all public databases (ENCODE, ChIP-Atlas, recount3) into comprehensive supplementary files and generates high-level training summary tables.

- **Input:** Per-DB prediction CSVs, metadata JSONs for all public databases, Comet-ML experiment archive
- **Output:** Comprehensive merged supplementary prediction files, training summary tables

---

## Marimo apps (`.py`)

Interactive [marimo](https://marimo.io) apps (reactive `.py` notebooks, not Jupyter). Run with `marimo edit <path>` (or `marimo run <path>` read-only). Listed by their location in the tree.

### `signal/mo_umap_plot.py`

Interactive UMAP explorer. Loads a precomputed pickled UMAP embedding, colours points by any metadata column, and lets you box/lasso-select points to inspect them in a table; a second panel re-projects the selection onto another embedding (e.g. standard vs densMAP).

- **Input:** `embedding_*_2D_*.pkl` (precomputed UMAP), metadata v2
- **Output:** Interactive (no files)

### `metadata/mo_celltype_similarity_groups.py`

Explores the within-cohort cell-type Lin-similarity matrix group by group. Pick a similarity threshold and a cell-type group and inspect it as a heatmap and an interactive node-link graph. Needs the `biospecimen_similarity` venv (has `nxontology`).

- **Input:** `celltype_similarity_matrix.csv` (from `evaluate_biospecimen_similarity.ipynb`); optional `*_sweep_groups.csv`
- **Output:** Interactive (no files)

### `predictions/mo_compare_rna_mapping_predictions.py`

Compares RNA-Seq mapping variants against the training Unique mapping, per classifier, per fold, to check robustness to the mapping change. Section 1 pairs Unique vs UniqueMultiple 1-to-1 on `(EpiRR, strand)` (UniqueMultiple never seen by the models), with a conformal-prediction follow-up on the new errors. A later section handles **Unstranded** RNA (plusRaw + minusRaw summed via `utils/preprocessing/sum_stranded_rna_hdf5.py`) as a 2-to-1 comparison: the averaged plus/minus Unique probabilities vs the single unstranded prediction.

- **Input:** `concatenated_test_prediction_*.csv` per classifier (with an `origin` column) under `RNA_UniqueMultiple/` and `RNA_Unstranded/`
- **Output:** Interactive comparison (tables + metrics box plots)

### `predictions/conformal/mo_conformal_report.py`

Extensive per-classifier conformal-prediction report over a 10-fold CV run: §1 marginal-coverage sanity, §2 per-class coverage/set-size/empty-rate, §3 Mondrian feasibility + marginal-vs-Mondrian, §4 RAPS/SAPS hyperparameter sensitivity, §5 cross-classifier "hands up" scan. RAPS/SAPS focus, LAC/APS faded refs. (See `utils/conformal/methods.md`.)

- **Input:** `split*/validation_prediction.csv` under a classifier run dir
- **Output:** Cached per-fold `conformal_report.csv`; summary PNGs to `<run>/conformal_report/`

### `predictions/conformal/mo_conformal_exploration.py`

Donor-sex conformal scratchpad — the original exploration notebook (marginal coverage, per-class breakdown, Mondrian comparison) that the report app generalized.

- **Input:** `split*/validation_prediction.csv`
- **Output:** Interactive (no files)

### `predictions/conformal/mo_conformal_cv_examination.py`

Training-data QC / mislabel flagging. READ-ONLY viewer of the precomputed within-fold leave-one-out sets: per-class flag composition (clean/hedge/disagree/empty), per-class coverage, marginal-vs-Mondrian where feasible, a flagged-sample table, and the flagged samples in UMAP/PCA space coloured by flag. Run `precompute --mode cv-examine` first.

- **Input:** `<run>/conformal_sets/cv_examination_*.csv` (from `python -m epiclass.utils.conformal.precompute --mode cv-examine`), metadata, UMAP/PCA embeddings
- **Output:** Interactive (no files)

### `predictions/conformal/mo_conformal_deployment.py`

Explore CV+ prediction sets on new data. READ-ONLY viewer: set-size distribution, a browsable prediction-set table, per-predicted-class breakdown, and the new samples in embedding space coloured by set size. Run `precompute --mode deploy` first.

- **Input:** `<data>/conformal_sets/cv_plus_sets_*.csv` (from `precompute --mode deploy`), metadata, UMAP/PCA embeddings
- **Output:** Interactive (no files)
