from scripts.eval_cqn_as_snapshot_sweep import parse_step_set


def test_parse_step_set_empty_means_no_allowlist():
    assert parse_step_set("") == set()


def test_parse_step_set_accepts_checkpoint_csv():
    assert parse_step_set("12500, 15000,17500,20000") == {
        12500,
        15000,
        17500,
        20000,
    }
