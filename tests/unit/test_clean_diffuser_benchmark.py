from benchmarks.compare_policy_backends import _comparable


def test_clean_diffuser_benchmark_rejects_unet_compatibility_mismatch():
    reference = {
        "operator_variant": "torch",
        "compatibility_mode": "clean_diffuser",
        "global_condition_embed_dim": 256,
    }

    candidate = dict(reference, operator_variant="legacy")
    comparable, mismatches = _comparable(candidate, reference)
    assert comparable is False
    assert any("operator_variant" in mismatch for mismatch in mismatches)

    candidate = dict(reference, global_condition_embed_dim=0)
    comparable, mismatches = _comparable(candidate, reference)
    assert comparable is False
    assert any("global_condition_embed_dim" in mismatch for mismatch in mismatches)


def test_clean_diffuser_benchmark_rejects_training_contract_mismatch():
    reference = {
        "optimizer": "adamw",
        "learning_rate": 1e-4,
        "weight_decay": 1e-2,
        "training_schedule": "cosine_discrete",
        "loss_reduction": "mse_mean",
        "sample_temperature": 1.0,
        "action_bounds": [-1.0, 1.0],
    }

    for field, mismatch_value in {
        "optimizer": "adam",
        "learning_rate": 3e-4,
        "weight_decay": 0.0,
        "training_schedule": "linear",
        "loss_reduction": "mse_sum",
        "sample_temperature": 0.5,
        "action_bounds": None,
    }.items():
        candidate = dict(reference, **{field: mismatch_value})
        comparable, mismatches = _comparable(candidate, reference)
        assert comparable is False
        assert any(field in mismatch for mismatch in mismatches)


def test_clean_diffuser_benchmark_rejects_missing_contract_metadata():
    comparable, mismatches = _comparable({}, {})

    assert comparable is False
    assert any("missing from jax, torch result" in mismatch for mismatch in mismatches)
