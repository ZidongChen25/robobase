from robobase.method.cqn_as import CQNAS


def test_exploration_checkpoint_helpers_allow_subclass_without_cqn_as_rngs():
    # CQNFlowAS historically initialized JaxRLMethodBase directly and did not
    # create CQNAS's later exploration-only runtime arrays.  Loading an older
    # evaluation snapshot must remain a no-op for those absent mechanisms.
    agent = object.__new__(CQNAS)

    assert agent._exploration_checkpoint_state() == {}
    agent._load_exploration_checkpoint_state(
        {
            "bin_flip_rng_state": {"ignored": True},
            "bin_explore_rng_state": {"ignored": True},
            "episodic_twin_head_rng_state": {"ignored": True},
        }
    )
    agent._resample_episodic_twin_heads([0])
    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["episodic_twin_head_assignments"] == 0.0
