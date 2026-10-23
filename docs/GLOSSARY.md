# Glossary

Terms as they are used in this codebase. Code identifiers are in `backticks`.

## Clinical / domain

| Term | Meaning here |
|------|--------------|
| **Neoadjuvant therapy** | Treatment (usually chemotherapy) given *before* surgery. The reports describe the tumor after this treatment and before surgery. |
| **Post-neoadjuvant, pre-surgery MRI** | Breast MRI report written after treatment and before the operation. Column `preop_MRI_text`. |
| **Pathology report** | Microscopic report on the resected tissue. Column `path_report_text`. |
| **Index side** | Which breast (`left`/`right`) the labels refer to. Column `index_side`. Approach 1 tells the LLM to focus on it; Approach 2 recoding does not use it. |
| **Dispersion / dispersiveness** | How spatially scattered the residual tumor is (scattered foci vs one compact mass). Ground truth column `dispersion_invasive_DCIS_geographic`, a continuous score (Approach 1 prompts constrain predictions to 0–450). How the score is measured is not documented in the code. |
| **High / low dispersion** | Binary label: high if score ≥ 85 (`DISPERSION_HIGH_THRESHOLD`, `DISPERSION_TRUE_HIGH_THRESHOLD`). |
| **Relapse** | Binary outcome, column `relapse` (1 = relapsing). |
| **DCIS** | Ductal carcinoma in situ: non-invasive cancer inside the ducts. |
| **Invasive carcinoma** | Cancer that has broken out of the ducts into surrounding tissue. |
| **LVI** | Lymphovascular invasion: tumor cells inside lymph or blood vessels. |
| **NME** | Non-mass enhancement: an MRI pattern of enhancement that is not a discrete mass, often linear, segmental or regional. |
| **BPE** | Background parenchymal enhancement: normal breast-tissue enhancement on MRI. A confounder. |
| **Multifocal / multicentric** | Several foci in one quadrant / foci in different quadrants. |
| **Tumor bed** | Region where the tumor was before treatment response. |
| **Margin** | Edge of the surgical specimen. "Close" or "positive" margins mean tumor near or at the edge. |
| **Modality / tier** | Which report(s) the model sees. Approach 1 tiers: `mri_only`, `pathology_only`, `mri_plus_pathology`. Approach 2 dataset keys: `mri`, `path`, `combined`, plus the derived MRI pathways. |
| **PHI** | Protected health information. See root `CLAUDE.md`. |

## LLM / NLP

| Term | Meaning here |
|------|--------------|
| **Stanford AI Sandbox / SecureGPT** | The institution's PHI-approved Azure-OpenAI gateway. Models `gpt-5-nano` (default) and `gpt-5`. |
| **Few-shot prompting** | Putting a few labeled examples in the prompt so the model imitates them. Approach 1 uses 4 (2 high, 2 low). |
| **Shot set** | A named choice of exemplar rows (`SHOT_SETS` in `approach1/config.py`). |
| **Exemplar** | One labeled example inside the prompt. Its rows are removed from the test set. |
| **Reasoning effort** | GPT-5 API parameter (`minimal`/`low`/`medium`/`high`). `none` omits it. |
| **Validation token / needle** | A synthetic string (`CTXCHK_…`) appended to each prompt. The model must echo it, which checks that it read to the end of the context. Reported as `needle_single_token_rate`. |
| **Key evidence** | Up to 6 verbatim quotes the Approach 1 model cites. They must be exact substrings of the supplied report(s), ≤ 35 words (25 targeted). |
| **Quote repair** | When a quote is not an exact substring, the code searches for the closest exact report span and substitutes it, or drops the quote (`schema/quote_helpers.py`). |
| **Extraction** | Approach 2's LLM step: from one report, list `seed_aligned_phrases` and `denovo_candidate_phrases` (quote, concept, polarity, certainty, …), plus `quantitative_attributes` and `report_level_summary`. |
| **Seed vs de novo phrase** | Seed: matches the guidance vocabulary. De novo: anything else the model thinks is relevant. |
| **Phrase slug** | Normalized identifier for an extracted quote (`make_slug(quote)`). One feature per slug in `phrase_binary`. |
| **Canonical group / concept** | One of the 16 `SHARED_CONCEPT_ONTOLOGY` concepts (e.g. `multiplicity`, `non_mass_enhancement`), or `other_candidate_feature`. Each has hand-written regex patterns. |
| **Ontology** | The fixed concept list plus regexes in `approach2/config.py`. Background knowledge, not learned. |
| **Polarity / negation / uncertainty** | Whether a finding is affirmed, negated ("no residual…") or hedged ("possible…"). Recoding detects these with regexes within ±80 characters of a match. |
| **Recoding** | Applying the frozen lexicon and ontology regexes to report text (train and test) to make numeric features. No LLM is involved. |
| **Representation** | Which feature columns a model uses: `group_binary`, `group_count`, `group_status`, `phrase_binary`, `weighted_concept_score`, `weighted_plus_group_status`. |
| **Early fusion** | Concatenating MRI and pathology feature columns into one matrix (`combined`). |

## Statistical / ML

| Term | Meaning here |
|------|--------------|
| **Outer split** | One train/test partition for honest evaluation. Default `repeated_mc`: 5 independent stratified 80/20 draws. |
| **Repeated Monte Carlo (MC) CV** | Repeated random train/test splits. Test sets can overlap and some cases are never tested. |
| **Stratified k-fold** | k disjoint folds; every case is tested exactly once. Stratified keeps the class ratio similar in each fold. |
| **Nested resampling** | Inner resampling (lexicon rediscovery, hyperparameter search) runs inside the outer-train set only. |
| **Rediscovery subsplit** | One of the inner subsets (default 25 MC subsamples at 80%) of outer-train cases used to measure phrase stability. |
| **Selection frequency** | Share of rediscovery subsplits where a phrase or group appears in ≥ `min_*_cases` (2) cases. |
| **Stability threshold** | Minimum selection frequency to enter the frozen lexicon (`--stability-threshold`). |
| **Frozen lexicon** | The stable phrase/group list fixed on outer-train, then applied unchanged to test. |
| **Leakage** | Any information from test cases influencing training. The design point of Approach 2 is to prevent it. |
| **GridSearchCV** | sklearn hyperparameter search with internal CV on the training data. |
| **Ridge / ElasticNet / Huber / PLS / SVR / SVM** | Regularized linear models. L2 (ridge) shrinks coefficients; ElasticNet mixes L1+L2 (can zero features); Huber is robust to outliers; PLS projects to latent components; SVR/SVM use margin-based losses. |
| **`LowInfoFeatureFilter`** | Drops columns that are almost always missing, never nonzero, or constant in the training fold. |
| **Teacher-student** | The MRI-only "student" model is trained to also predict a pathology-based "teacher" score and pathology concepts, so MRI features learn pathology-informed directions. Test-time inputs are MRI only. |
| **Pathology calibration (pathcal)** | Weighting MRI concepts by how reliably they co-occur with pathology concepts in training cases (`calibration.py`). Unrelated to probability calibration. |
| **Deduplication (dedup)** | Averaging a case's predictions across the repeated MC test sets it appeared in, giving one prediction per case before metrics. |
| **MAE / RMSE** | Mean absolute error / root mean squared error of the dispersion score. |
| **Spearman ρ** | Rank correlation between predicted and true score. |
| **AUROC** | Probability that a random positive case is ranked above a random negative one. 0.5 = chance. |
| **AUPRC** | Area under the precision-recall curve. The no-skill baseline equals prevalence. |
| **Brier score** | Mean squared error of predicted probabilities. Only meaningful for real probabilities. |
| **Calibration intercept / slope** | Logistic recalibration of predicted probabilities. Slope < 1 suggests overconfident probabilities. |
| **Sensitivity / specificity / PPV / NPV** | True-positive rate / true-negative rate / precision / negative predictive value. |
| **Balanced accuracy** | Mean of sensitivity and specificity. |
| **Bootstrap percentile CI** | Resample cases with replacement (default 1000×), recompute the metric, take the 2.5th/97.5th percentiles. |
| **Permutation test** | Shuffle labels many times to build a null distribution for AUROC/AUPRC. Used for relapse in Approach 2. |
| **Smoothed odds ratio** | Odds ratio with +0.5 added to counts. Used in Approach 1 evidence attribution and Approach 2 reliability. |
| **Lift / Δp / mutual information** | Co-occurrence statistics between an MRI concept and a pathology concept (`compute_cross_modal_reliability`). |
