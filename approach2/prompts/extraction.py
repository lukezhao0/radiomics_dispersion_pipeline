"""Extraction prompt templates and ontology guidance."""

from __future__ import annotations

SHARED_ONTOLOGY_GUIDANCE = """
SHARED BIOLOGICAL CONCEPT ONTOLOGY FOR DISPERSION-RELEVANT LANGUAGE

Extract phrases that may map to one or more of these concept names. Use these
names in candidate_concepts when appropriate. Do not force a concept if the text
does not support it.

- extent_span: long span, large extent, broad area, disease/tumor bed spanning a dimension
- multiplicity: multiple foci, multifocality, several residual sites
- multicentricity_separate_sites: separate quadrants/regions/sites, multicentric disease
- distribution_linear_segmental_regional: linear, segmental, ductal, regional distribution
- fragmentation_scattered_patchy_discontinuous: patchy, scattered, discontinuous, skip-like pattern
- residual_tumor_presence: residual enhancement or residual viable carcinoma/disease
- non_mass_enhancement: NME or non-mass enhancement pattern
- invasive_disease: invasive carcinoma/disease component
- in_situ_disease_dcis: DCIS / ductal carcinoma in situ / in-situ component
- treatment_response: complete, near-complete, partial, poor, decreased enhancement/cellularity
- treatment_effect_tumor_bed: treatment effect, tumor bed, fibrosis, therapy-related changes
- localized_compact_residual: single focal mass/focus, localized/compact residual disease
- diffuse_scattered_residual: diffuse or scattered residual disease/enhancement
- lymphovascular_invasion: LVI / lymphovascular invasion
- margin_proximity: margin, close/positive margin, distance from margin
- benign_or_nonspecific_enhancement: nonspecific, background, probably benign enhancement
""".strip()

SEED_GUIDANCE = f"""
DISPERSIVENESS SEED GUIDANCE (USE ONLY AS INITIAL CUEING, NOT AS A CLOSED VOCABULARY)

Concept families of interest include:
- spatial scatter / scattered foci / satellites
- multifocality / multicentricity
- discontinuity / separated foci / patchy or discontinuous disease
- infiltrative spread / irregular infiltrative residual disease
- broad extent / long span / large area involved
- localization / compact single residual focus
- minimal residual disease / near-complete response / no substantial residual disease

MRI-oriented examples:
- non-mass enhancement
- clumped / segmental / linear / regional enhancement
- patchy enhancement
- diffuse residual enhancement rather than one compact mass
- broad extent of abnormal enhancement
- scattered enhancing foci or satellites
- possible, nonspecific, indeterminate, favored benign, or treatment-related enhancement
- comparison-to-prior response language such as decreased, resolved, persistent, residual

Pathology-oriented examples:
- multiple residual invasive foci
- discontinuous residual carcinoma
- extensive residual DCIS
- lymphovascular invasion
- satellites / separate microscopic foci
- close margins / broad span of disease
- minimal residual disease / focal residual disease

{SHARED_ONTOLOGY_GUIDANCE}
""".strip()

SYSTEM_MSG = (
    "You are an advanced, careful clinical NLP model operating in a PHI-secure environment. "
    "Your task is lexical feature discovery. "
    "Use only the provided report text. Do not invent content not present in the report. "
    "Return valid JSON only. "
    "For every extracted phrase, copy an exact quote from the report text. "
    "Do not reveal hidden chain-of-thought. Provide only the requested structured fields."
)
