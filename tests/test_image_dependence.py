import pytest
from PIL import Image

from evaluate import _clustered_accuracy_ci, _dataset_image_cluster_ids
from scripts.analyze_image_dependence import attach_image_cluster_ids, paired_summary


def _rows(correctness, image_ids=None):
    result = {}
    for index, correct in enumerate(correctness):
        row = {"row_index": index, "target_index": 1, "correct": correct}
        if image_ids is not None:
            row["image_cluster_id"] = image_ids[index]
        result[index] = row
    return result


def test_cluster_bootstrap_keeps_questions_from_an_image_together():
    actual = _rows([True, True, False, False], ["image-a", "image-a", "image-b", "image-b"])
    blank = _rows([False, True, False, True], ["image-a", "image-a", "image-b", "image-b"])

    result = paired_summary(actual, blank, bootstrap_samples=200, seed=7)

    assert result["n"] == 4
    assert result["image_cluster_count"] == 2
    assert result["image_cluster_uncertainty_available"] is True
    assert len(result["image_cluster_bootstrap_95ci_percentage_points"]) == 2
    assert result["image_cluster_sign_test_non_tied_clusters"] == 2
    assert result["image_cluster_sign_test_two_sided_p"] == 1.0


def test_cluster_uncertainty_is_marked_unavailable_for_legacy_artifacts():
    actual = _rows([True, False])
    blank = _rows([False, False])

    result = paired_summary(actual, blank, bootstrap_samples=100, seed=0)

    assert result["image_cluster_uncertainty_available"] is False
    assert result["image_cluster_bootstrap_95ci_percentage_points"] is None


def test_paired_summary_rejects_cluster_mismatch():
    actual = _rows([True], ["image-a"])
    blank = _rows([False], ["image-b"])

    with pytest.raises(ValueError, match="Image cluster mismatch"):
        paired_summary(actual, blank, bootstrap_samples=100, seed=0)


def test_clustered_accuracy_ci_point_estimate_and_degenerate_clusters():
    # Two clusters of two QA rows each; overall accuracy is 50%.
    correct = [True, False, True, False]
    clusters = ["a", "a", "b", "b"]

    result = _clustered_accuracy_ci(correct, clusters, bootstrap_samples=200, seed=3)

    assert result["point_estimate"] == pytest.approx(50.0)
    assert result["cluster_count"] == 2
    assert result["resampling_unit"].startswith("image cluster")
    # Both clusters have identical accuracy, so every resample returns 50%.
    assert result["lower"] == pytest.approx(50.0)
    assert result["upper"] == pytest.approx(50.0)


def test_clustered_accuracy_ci_resamples_images_not_questions():
    # Cluster "a" has four correct rows, cluster "b" four wrong rows. Row-level
    # bootstrap would produce a much narrower interval than the two-cluster
    # bootstrap, which must sometimes draw only one of the two clusters.
    correct = [True] * 4 + [False] * 4
    clusters = ["a"] * 4 + ["b"] * 4

    result = _clustered_accuracy_ci(correct, clusters, bootstrap_samples=500, seed=0)

    assert result["point_estimate"] == pytest.approx(50.0)
    assert result["lower"] == pytest.approx(0.0)
    assert result["upper"] == pytest.approx(100.0)


def test_clustered_accuracy_ci_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        _clustered_accuracy_ci([True], ["a", "b"])


def test_dataset_image_cluster_ids_are_content_stable():
    class TinyDataset:
        data = [
            {"image": Image.new("RGB", (3, 3), color=(10, 20, 30))},
            {"image": Image.new("RGB", (3, 3), color=(10, 20, 30))},
            {"image": Image.new("RGB", (3, 3), color=(30, 20, 10))},
        ]

        def __len__(self):
            return len(self.data)

    cluster_ids = _dataset_image_cluster_ids(TinyDataset())

    assert cluster_ids[0] == cluster_ids[1]
    assert cluster_ids[0] != cluster_ids[2]


def test_attach_cluster_ids_aligns_each_condition_by_row_index():
    conditions = {
        "actual": {0: {"row_index": 0}, 1: {"row_index": 1}},
        "blank": {0: {"row_index": 0}, 1: {"row_index": 1}},
    }

    attach_image_cluster_ids(conditions, ["image-a", "image-b"])

    assert conditions["actual"][0]["image_cluster_id"] == "image-a"
    assert conditions["blank"][1]["image_cluster_id"] == "image-b"
