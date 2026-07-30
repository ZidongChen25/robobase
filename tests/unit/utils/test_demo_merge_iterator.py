import numpy as np

import robobase.utils as utils


def test_demo_merged_iterator_supports_numpy_without_torch(monkeypatch):
    monkeypatch.setattr(utils, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(utils, "torch", None)

    replay_iter = iter(
        [
            {
                "obs": np.zeros((2, 3), dtype=np.float32),
                "demo": np.zeros((2,), dtype=np.float32),
            }
        ]
    )
    demo_replay_iter = iter(
        [
            {
                "obs": np.ones((1, 3), dtype=np.float32),
                "demo": np.zeros((1,), dtype=np.float32),
            }
        ]
    )

    batch = next(utils.merge_replay_demo_iter(replay_iter, demo_replay_iter))

    np.testing.assert_array_equal(
        batch["obs"],
        np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(batch["demo"], np.array([0, 0, 1], dtype=np.float32))
