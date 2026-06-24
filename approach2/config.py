"""Constants, ontology definitions, and shared column metadata for approach2."""

from __future__ import annotations

from typing import Any, Dict

DISPERSION_TRUE_HIGH_THRESHOLD = 85.0
RANDOM_SEED = 17
INNER_CV_MAX_SPLITS = 5
COEF_ZERO_TOL = 1e-8
EPS = 1e-12
DEFAULT_BOOTSTRAP_N = 1000

TARGET_NAME_DISPERSION_SCORE = "dispersion_score"
TARGET_NAME_DISPERSION_HIGH_LOW = "dispersion_high_low"
TARGET_NAME_RELAPSE_STATUS = "relapse_status"

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\bnegative for\b",
    r"\bfree of\b",
    r"\bneither\b",
    r"\bnone\b",
    r"\bresolved\b",
    r"\bresolution of\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bprobable\b",
    r"\blikely\b",
    r"\bsuspicious\b",
    r"\bsuspected\b",
    r"\bcannot exclude\b",
    r"\bmay represent\b",
    r"\bindeterminate\b",
    r"\buncertain\b",
    r"\bfavored\b",
    r"\bprobably\b",
    r"\bpresumably\b",
]

# Shared ontology. The concept names are intentionally stable identifiers used
# throughout recoding, reliability analysis, weighting, and interpretation.
SHARED_CONCEPT_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "extent_span": {
        "definition": "Long span, large extent, or broad area of abnormality/tumor bed/disease.",
        "category": "spatial_morphology_response",
        "mri_examples": "spanning 6 cm; extent of enhancement; large area of NME",
        "path_examples": "residual disease spanning; tumor bed measures; extent of DCIS",
        "patterns": [r"\bextent\b", r"\bspan\b", r"\bspanning\b", r"\bmeasur", r"\blarge area\b", r"\bbroad\b", r"\bdiffuse\b", r"\bextensive\b", r"\bcm\b", r"\bmm\b"],
    },
    "multiplicity": {
        "definition": "Multiple foci/sites suggesting more than one localized residual focus.",
        "category": "spatial_morphology_response",
        "mri_examples": "multiple foci; multifocal enhancement; satellite foci",
        "path_examples": "multiple residual foci; multifocal residual carcinoma",
        "patterns": [r"\bmultifocal\b", r"\bmultiple\b", r"\bseveral\b", r"\bsatellite", r"\badditional foci\b", r"\bseparate foci\b", r"\bsmall foci\b"],
    },
    "multicentricity_separate_sites": {
        "definition": "Disease or abnormality in separate regions/sites/quadrants.",
        "category": "spatial_morphology_response",
        "mri_examples": "multicentric disease; separate sites",
        "path_examples": "separate foci; widely separated residual disease",
        "patterns": [r"\bmulticentric\b", r"\bseparate sites\b", r"\bseparate areas\b", r"\bwidely separated\b", r"\bseparated by\b", r"\bdifferent quadrant"],
    },
    "distribution_linear_segmental_regional": {
        "definition": "Linear, segmental, ductal, or regional distribution pattern.",
        "category": "spatial_morphology_response",
        "mri_examples": "segmental NME; linear enhancement; ductal distribution",
        "path_examples": "ductal distribution; regional spread; geographic disease",
        "patterns": [r"\bsegmental\b", r"\blinear\b", r"\bregional\b", r"\bductal distribution\b", r"\bclumped\b", r"\bgeographic\b"],
    },
    "fragmentation_scattered_patchy_discontinuous": {
        "definition": "Patchy, scattered, discontinuous, or fragmented residual abnormality/disease.",
        "category": "spatial_morphology_response",
        "mri_examples": "patchy enhancement; scattered foci; discontinuous enhancement",
        "path_examples": "scattered microscopic foci; discontinuous residual carcinoma",
        "patterns": [r"\bdiscontinuous\b", r"\bdiscontinuity\b", r"\bpatchy\b", r"\bskip\b", r"\bscattered\b", r"\bfragment", r"\bintermixed\b"],
    },
    "residual_tumor_presence": {
        "definition": "Residual disease/tumor or residual enhancement after neoadjuvant therapy.",
        "category": "residual_disease",
        "mri_examples": "residual enhancement; persistent NME",
        "path_examples": "residual viable carcinoma; residual invasive carcinoma; residual DCIS",
        "patterns": [r"\bresidual\b", r"\bpersistent\b", r"\bviable carcinoma\b", r"\bresidual carcinoma\b", r"\bresidual disease\b", r"\bremaining\b"],
    },
    "non_mass_enhancement": {
        "definition": "MRI non-mass enhancement pattern, including NME abbreviation.",
        "category": "spatial_morphology_response",
        "mri_examples": "non-mass enhancement; NME",
        "path_examples": "not usually literal, may align to broad/ductal/DCIS concepts",
        "patterns": [r"\bnon[- ]mass enhancement\b", r"\bnme\b", r"\bnon mass\b"],
    },
    "invasive_disease": {
        "definition": "Invasive carcinoma/disease component.",
        "category": "pathology_specific",
        "mri_examples": "often indirect, e.g. enhancing mass suspicious for invasive disease",
        "path_examples": "residual invasive carcinoma; invasive ductal carcinoma",
        "patterns": [r"\binvasive\b", r"\binvasion\b", r"\binfiltrating carcinoma\b"],
    },
    "in_situ_disease_dcis": {
        "definition": "DCIS or in-situ component/burden.",
        "category": "pathology_specific",
        "mri_examples": "NME in ductal distribution, often indirect",
        "path_examples": "DCIS; ductal carcinoma in situ; extensive intraductal component",
        "patterns": [r"\bdcis\b", r"\bductal carcinoma in situ\b", r"\bin situ\b", r"\bintraductal\b"],
    },
    "treatment_response": {
        "definition": "Language describing response magnitude after therapy.",
        "category": "spatial_morphology_response",
        "mri_examples": "decreased enhancement; near complete imaging response",
        "path_examples": "treatment effect; residual cellularity; partial response",
        "patterns": [r"\bdecrease\b", r"\bdecreased\b", r"\binterval decrease\b", r"\bshrink", r"\breduced\b", r"\bsmaller\b", r"\bresponse\b", r"\bcomplete response\b", r"\bnear complete\b", r"\bpartial response\b"],
    },
    "treatment_effect_tumor_bed": {
        "definition": "Treatment effect, tumor bed, fibrosis/scarring, or therapy-related changes.",
        "category": "response_treatment_effect",
        "mri_examples": "post-treatment change; treatment-related enhancement",
        "path_examples": "tumor bed; treatment effect; fibrosis; residual cellularity",
        "patterns": [r"\btreatment effect\b", r"\btumou?r bed\b", r"\bfibrosis\b", r"\btherapy[- ]related\b", r"\bpost[- ]treatment\b", r"\bcellularity\b"],
    },
    "localized_compact_residual": {
        "definition": "Single, focal, compact, localized residual abnormality/disease.",
        "category": "spatial_morphology_response",
        "mri_examples": "single residual mass; focal enhancement",
        "path_examples": "single focus; localized residual carcinoma",
        "patterns": [r"\blocalized\b", r"\bcompact\b", r"\bsingle\b", r"\bsolitary\b", r"\bdominant mass\b", r"\bfocal\b", r"\bsingle residual\b", r"\bone focus\b"],
    },
    "diffuse_scattered_residual": {
        "definition": "Diffuse, scattered, broad, or patchy residual disease/enhancement.",
        "category": "spatial_morphology_response",
        "mri_examples": "diffuse residual enhancement; broad NME",
        "path_examples": "scattered residual disease; widely distributed residual tumor",
        "patterns": [r"\bdiffuse residual\b", r"\bpatchy residual\b", r"\bscattered residual\b", r"\bmultiple residual\b", r"\bdiscontinuous residual\b", r"\bwidely distributed\b"],
    },
    "lymphovascular_invasion": {
        "definition": "Lymphovascular invasion.",
        "category": "pathology_specific",
        "mri_examples": "not directly visible in routine report language",
        "path_examples": "lymphovascular invasion; LVI",
        "patterns": [r"\blymphovascular invasion\b", r"\blvi\b"],
    },
    "margin_proximity": {
        "definition": "Margin proximity/positivity, a pathology-localization and residual extent signal.",
        "category": "pathology_specific",
        "mri_examples": "not usually direct; relationship to surgical margins may be absent",
        "path_examples": "margin; mm from margin; close margin; positive margin",
        "patterns": [r"\bmargin\b", r"\bmm from\b", r"\bclose margin\b", r"\bpositive margin\b", r"\bclear margin\b"],
    },
    "benign_or_nonspecific_enhancement": {
        "definition": "MRI enhancement described as nonspecific, probably benign, or background-like.",
        "category": "ambiguity_noise",
        "mri_examples": "background parenchymal enhancement; probably benign; nonspecific enhancement",
        "path_examples": "not direct pathology correlate; used as ambiguity penalty",
        "patterns": [r"\bnonspecific\b", r"\bprobably benign\b", r"\bfavored benign\b", r"\bbackground parenchymal enhancement\b", r"\bbpe\b", r"\bbenign enhancement\b"],
    },
}

CANONICAL_GROUP_PATTERNS = {k: v["patterns"] for k, v in SHARED_CONCEPT_ONTOLOGY.items()}

SPATIAL_MORPH_RESPONSE_GROUPS = {
    k for k, v in SHARED_CONCEPT_ONTOLOGY.items()
    if v["category"] in {"spatial_morphology_response", "residual_disease", "response_treatment_effect"}
}
DISTRIBUTION_GROUPS = {
    "multiplicity",
    "multicentricity_separate_sites",
    "distribution_linear_segmental_regional",
    "fragmentation_scattered_patchy_discontinuous",
    "diffuse_scattered_residual",
    "non_mass_enhancement",
}
AMBIGUITY_GROUPS = {"benign_or_nonspecific_enhancement"}

META_COLS = {
    "case_id",
    "row_index",
    "dispersion_true",
    "dispersion_true_high_low",
    "relapse_true",
    "split_id",
    "split_role",
    "report_mode",
}
