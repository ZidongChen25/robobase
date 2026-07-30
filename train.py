from pathlib import Path
import logging
import os

import hydra


def _configure_backend_logging(cfg):
    backend_cfg = cfg.get("backend", None)
    if backend_cfg is None:
        return
    if str(backend_cfg.get("name", "torch")).lower() != "jax":
        return
    if not bool(backend_cfg.get("suppress_cpp_logs", True)):
        return

    # Silence noisy XLA/JAX C++ logs such as cuda_timer.cc timing warnings.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


@hydra.main(
    config_path="robobase/cfgs", config_name="robobase_config", version_base=None
)
def main(cfg):
    from robobase.gpu import apply_requested_gpu

    apply_requested_gpu(cfg)
    _configure_backend_logging(cfg)

    from robobase.workspace import Workspace

    root_dir = Path.cwd()

    workspace = Workspace(cfg)

    latest_snapshot = workspace.work_dir / "snapshots" / "latest_snapshot.pkl"
    legacy_snapshot = root_dir / "snapshot.pt"
    if latest_snapshot.exists():
        print(f"resuming: {latest_snapshot}")
        workspace.load_snapshot(latest_snapshot)
    elif legacy_snapshot.exists():
        print(f"resuming: {legacy_snapshot}")
        workspace.load_snapshot(legacy_snapshot)
    workspace.train()


if __name__ == "__main__":
    main()
