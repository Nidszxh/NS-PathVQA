"""Unit tests for AnatomicalOntology and DatasetAdapters (PathVQA, VQA-RAD, SLAKE)."""

from data.dataset_adapter import (
    AnatomicalOntology,
    PathVQAAdapter,
    SLAKEAdapter,
    VQARADAdapter,
)


def test_anatomical_ontology_hierarchy():
    ontology = AnatomicalOntology()
    parents_cortex = ontology.get_parent_systems("renal cortex")
    assert "urinary system" in parents_cortex

    parents_stomach = ontology.get_parent_systems("gastric")
    assert "gastrointestinal system" in parents_stomach

    # Synonyms and relations
    assert ontology.normalize_term("renal") == "kidney"
    assert ontology.normalize_term("hepatic") == "liver"
    assert ontology.are_related("gastric", "colon") is True
    assert ontology.are_related("renal cortex", "urinary system") is True
    assert ontology.are_related("lung", "brain") is False


def test_pathvqa_adapter():
    adapter = PathVQAAdapter()
    vocab = ["gastrointestinal system", "lung", "renal cortex", "red", "yes", "no", "malignant"]
    regions = adapter.discover_regions(vocab)
    assert "gastrointestinal system" in regions
    assert "lung" in regions
    assert "renal cortex" in regions
    assert "yes" not in regions
    assert "red" not in regions

    report = adapter.get_coverage_report(vocab, regions)
    assert report["total_answers"] == len(vocab)
    assert report["linked_count"] >= 3
    assert report["coverage"] > 0.0


def test_vqarad_and_slake_adapters():
    rad_adapter = VQARADAdapter()
    slake_adapter = SLAKEAdapter()

    rad_vocab = ["chest", "brain", "pleura", "normal", "yes", "no"]
    rad_regions = rad_adapter.discover_regions(rad_vocab)
    assert "chest" in rad_regions
    assert "pleura" in rad_regions

    slake_vocab = ["liver", "rib", "trachea", "abnormal"]
    slake_regions = slake_adapter.discover_regions(slake_vocab)
    assert "liver" in slake_regions
    assert "trachea" in slake_regions
