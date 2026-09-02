import numpy as np
import pytest

from tools.compare_clahe_methods import (
    METHOD_ADAPTIVE,
    METHOD_FIXED,
    apply_fixed_clahe,
    build_candidate_mappings,
    discover_images,
    ensure_no_prior_unblinding,
    initialize_config,
    load_candidate_cache,
    load_roi_record,
    preferred_method,
    sha256_file,
    summarize_records,
    unblind_records,
    validate_candidate_cache_files,
    validate_masked_records,
    whole_image_clip_from_stats,
    write_candidate_cache,
    write_roi_record,
)


def test_fixed_clahe_preserves_shape_and_dtype():
    image = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
    result = apply_fixed_clahe(image, clip_limit=2.0)

    assert result.shape == image.shape
    assert result.dtype == np.uint8


@pytest.mark.parametrize("clip_limit", [0, -1, np.nan, np.inf])
def test_fixed_clahe_rejects_invalid_clip_limit(clip_limit):
    image = np.zeros((16, 16), dtype=np.uint8)

    with pytest.raises(ValueError, match="finite and positive"):
        apply_fixed_clahe(image, clip_limit=clip_limit)


def test_whole_image_clip_uses_known_low_entropy_branch():
    image = np.zeros((16, 16), dtype=np.uint8)

    assert whole_image_clip_from_stats(image) == 4.8


def test_whole_image_clip_uses_known_high_entropy_branch():
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)

    assert whole_image_clip_from_stats(image) == 3.6


def test_candidate_mapping_is_deterministic_complete_and_counterbalanced():
    source_hashes = [f"{index:064x}" for index in range(7)]
    first = build_candidate_mappings(source_hashes, 1729, "reviewer_1")
    second = build_candidate_mappings(source_hashes, 1729, "reviewer_1")

    assert first == second
    assert set(first) == set(source_hashes)
    assert all(
        {method_a, method_b} == {METHOD_ADAPTIVE, METHOD_FIXED}
        for method_a, method_b, _block_index in first.values()
    )
    adaptive_on_a = sum(
        method_a == METHOD_ADAPTIVE
        for method_a, _method_b, _block_index in first.values()
    )
    assert abs(adaptive_on_a - (len(source_hashes) - adaptive_on_a)) <= 1


def test_candidate_mapping_rejects_duplicate_hashes():
    with pytest.raises(ValueError, match="must be unique"):
        build_candidate_mappings(["a" * 64, "a" * 64], 1729, "reviewer_1")


def test_discovery_accepts_uppercase_tiff_suffix(tmp_path):
    image_path = tmp_path / "example.TIF"
    image_path.write_bytes(b"placeholder")

    images, common_root = discover_images([tmp_path])

    assert images == [image_path.resolve()]
    assert common_root == tmp_path.resolve()


def test_reviewer_choice_is_unmasked_from_recorded_mapping():
    mapping = (METHOD_FIXED, METHOD_ADAPTIVE)

    assert preferred_method("candidate_a", mapping) == METHOD_FIXED
    assert preferred_method("candidate_b", mapping) == METHOD_ADAPTIVE
    assert preferred_method("tie", mapping) == "tie"


def _masked_record(
    source_hash,
    choice="candidate_a",
    status="reviewed",
    exclusion_stage=None,
):
    if exclusion_stage is None:
        exclusion_stage = "post_exposure" if choice == "excluded" else ""
    return {
        "Source_SHA256": source_hash,
        "Reviewer_ID": "reviewer_1",
        "Reviewer_Choice": choice,
        "Status": status,
        "Exclusion_Stage": exclusion_stage,
        "ROI_X_Min_px": "0",
        "ROI_X_Max_px": "16",
        "ROI_Y_Min_px": "0",
        "ROI_Y_Max_px": "16",
        "Image_Width_px": "16",
        "Image_Height_px": "16",
        "Common_NLM_h": "8",
        "Fixed_Clip_Limit": "2.8",
        "Candidate_Cache_SHA256": "c" * 64,
        "Random_Seed": "1729",
        "Displayed_Method_Labels": "False",
    }


def test_masked_record_validation_rejects_duplicate_sources():
    source_hash = "a" * 64
    records = [_masked_record(source_hash), _masked_record(source_hash)]

    with pytest.raises(ValueError, match="Duplicate Source_SHA256"):
        validate_masked_records(records, "reviewer_1", {source_hash})


def test_masked_record_validation_rejects_unknown_choice():
    source_hash = "a" * 64
    records = [_masked_record(source_hash, choice="adaptive")]

    with pytest.raises(ValueError, match="Unknown Reviewer_Choice"):
        validate_masked_records(records, "reviewer_1", {source_hash})


def test_masked_record_validation_rejects_inconsistent_exclusion():
    source_hash = "a" * 64
    records = [_masked_record(source_hash, choice="excluded", status="reviewed")]

    with pytest.raises(ValueError, match="Inconsistent Status"):
        validate_masked_records(records, "reviewer_1", {source_hash})


def test_pre_exposure_exclusion_requires_no_roi_or_candidates():
    source_hash = "a" * 64
    record = _masked_record(
        source_hash,
        choice="excluded",
        status="excluded",
        exclusion_stage="pre_exposure",
    )
    for field in (
        "ROI_X_Min_px",
        "ROI_X_Max_px",
        "ROI_Y_Min_px",
        "ROI_Y_Max_px",
        "Common_NLM_h",
        "Fixed_Clip_Limit",
        "Candidate_Cache_SHA256",
    ):
        record[field] = ""

    validate_masked_records([record], "reviewer_1", {source_hash})


@pytest.mark.parametrize("stored_value", [False, "False"])
def test_masked_record_validation_accepts_boolean_and_csv_text(stored_value):
    source_hash = "a" * 64
    record = _masked_record(source_hash)
    record["Displayed_Method_Labels"] = stored_value

    validate_masked_records(
        [record],
        "reviewer_1",
        {source_hash},
        expected_show_labels=False,
    )


def test_unblinding_uses_counterbalanced_mapping_only_after_collection():
    source_hash = "a" * 64
    masked = [_masked_record(source_hash)]
    mappings = {source_hash: (METHOD_FIXED, METHOD_ADAPTIVE, 0)}

    result = unblind_records(masked, mappings)

    assert "Preferred_Method" not in masked[0]
    assert result[0]["Candidate_A_Method"] == METHOD_FIXED
    assert result[0]["Candidate_B_Method"] == METHOD_ADAPTIVE
    assert result[0]["Preferred_Method"] == METHOD_FIXED


def test_candidate_cache_round_trip_preserves_exact_arrays(tmp_path):
    path = tmp_path / "candidate.npz"
    adaptive = np.arange(64, dtype=np.uint8).reshape(8, 8)
    fixed = np.flipud(adaptive)
    adaptive_skeleton = adaptive % 2
    fixed_skeleton = fixed % 2
    metadata = {"source_sha256": "a" * 64, "common_nlm_h": 8}

    write_candidate_cache(
        path,
        metadata,
        adaptive,
        fixed,
        adaptive_skeleton,
        fixed_skeleton,
    )
    loaded_metadata, enhanced, skeletons = load_candidate_cache(
        path,
        "a" * 64,
        (8, 8),
    )

    assert loaded_metadata == metadata
    assert np.array_equal(enhanced[METHOD_ADAPTIVE], adaptive)
    assert np.array_equal(enhanced[METHOD_FIXED], fixed)
    assert np.array_equal(skeletons[METHOD_ADAPTIVE], adaptive_skeleton)
    assert np.array_equal(skeletons[METHOD_FIXED], fixed_skeleton)


def test_roi_record_round_trip_preserves_pre_exposure_selection(tmp_path):
    path = tmp_path / "roi.json"
    write_roi_record(path, (1, 7, 2, 8), (10, 12))

    assert load_roi_record(path, (10, 12)) == (1, 7, 2, 8)


def test_resume_rejects_records_without_matching_config(tmp_path):
    records_path = tmp_path / "clahe_masked_reviews.csv"
    records_path.write_text("orphaned records", encoding="utf-8")

    with pytest.raises(RuntimeError, match="without their configuration"):
        initialize_config(
            tmp_path / "clahe_comparison_config.json",
            records_path,
            settings={"reviewer_id": "reviewer_1"},
            input_manifest=[],
            implementation={"script": "hash-a"},
            environment={"python": "3.12"},
        )


def test_resume_rejects_changed_implementation_fingerprint(tmp_path):
    config_path = tmp_path / "clahe_comparison_config.json"
    records_path = tmp_path / "clahe_masked_reviews.csv"
    arguments = {
        "path": config_path,
        "records_path": records_path,
        "settings": {"reviewer_id": "reviewer_1"},
        "input_manifest": [{"source_sha256": "a" * 64}],
        "environment": {"python": "3.12"},
    }
    initialize_config(
        implementation={"script": "hash-a"},
        **arguments,
    )

    with pytest.raises(RuntimeError, match="implementation changed"):
        initialize_config(
            implementation={"script": "hash-b"},
            **arguments,
        )


def test_incomplete_resume_rejects_existing_unblinded_results(tmp_path):
    (tmp_path / "clahe_results_unblinded.csv").write_text(
        "previous results",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unblinded artifacts"):
        ensure_no_prior_unblinding(tmp_path, batch_complete=False)


def test_completed_record_detects_changed_candidate_cache(tmp_path):
    source_hash = "a" * 64
    cache_path = tmp_path / f"{source_hash[:12]}.npz"
    cache_path.write_bytes(b"first candidate bytes")
    record = _masked_record(source_hash)

    record["Candidate_Cache_SHA256"] = sha256_file(cache_path)
    validate_candidate_cache_files([record], tmp_path)

    cache_path.write_bytes(b"changed candidate bytes")
    with pytest.raises(RuntimeError, match="cache hash changed"):
        validate_candidate_cache_files([record], tmp_path)


def test_summary_counts_nonexcluded_records_and_known_fraction():
    records = [
        {"Preferred_Method": METHOD_ADAPTIVE},
        {"Preferred_Method": METHOD_ADAPTIVE},
        {"Preferred_Method": METHOD_ADAPTIVE},
        {"Preferred_Method": METHOD_FIXED},
        {"Preferred_Method": "tie"},
        {"Preferred_Method": "neither"},
        {"Preferred_Method": "excluded"},
    ]

    summary = summarize_records(records)

    assert summary["n_total_records"] == 7
    assert summary["n_reviewed_nonexcluded"] == 6
    assert summary["n_adaptive_preferred"] == 3
    assert summary["n_whole_image_preferred"] == 1
    assert summary["n_equivalent"] == 1
    assert summary["n_neither"] == 1
    assert summary["n_excluded"] == 1
    assert summary["n_directional_preferences"] == 4
    assert summary["adaptive_fraction_among_directional_preferences"] == 0.75
