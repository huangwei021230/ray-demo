"""Minimal real-Ray demo: distributed SGD fitting y = 2x + 1.

Run:
    .venv-ray/bin/python ray_real_demo.py

Then open the dashboard URL printed at startup (default http://127.0.0.1:8265).
The script trains slowly on purpose so you have time to browse Jobs / Actors /
Tasks / Cluster tabs while it runs.
"""

import os
import time

# Avoid auto-attaching to any pre-existing cluster on this host.
os.environ.pop("RAY_ADDRESS", None)

import ray

# --- tiny dataset: 20 points of y = 2x + 1 with x in [0, 1] --------------
DATA = [(x / 19.0, 2.0 * (x / 19.0) + 1.0) for x in range(20)]
NUM_WORKERS = 4
SHARDS = [DATA[i::NUM_WORKERS] for i in range(NUM_WORKERS)]


@ray.remote(num_cpus=1)
class Worker:
    """Holds one shard of data; computes gradient for given (w, b)."""

    def __init__(self, worker_id: int, shard):
        self.worker_id = worker_id
        self.shard = shard

    def compute_grad(self, w: float, b: float):
        # Slow it down so the dashboard is interesting to watch.
        time.sleep(0.5)
        dw = db = 0.0
        for x, y in self.shard:
            err = (w * x + b) - y
            dw += 2 * err * x
            db += 2 * err
        n = len(self.shard)
        return dw / n, db / n, self.worker_id


@ray.remote(num_cpus=1)
def average_grads(*grads):
    # A separate task so it shows up in the dashboard's task timeline.
    # Variadic so Ray dereferences each ObjectRef before passing in.
    time.sleep(0.2)
    dw = sum(g[0] for g in grads) / len(grads)
    db = sum(g[1] for g in grads) / len(grads)
    return dw, db


def main():
    # Ray's plasma socket path has a 107-byte cap. The project path is too
    # long, so we go through ~/.rd -> <project>/ray_session (symlink).
    # Real storage still lives in the project directory.
    short_temp = os.path.expanduser("~/.rd")
    assert os.path.islink(short_temp), (
        f"{short_temp} must be a symlink to the project's ray_session/. "
        "Run: ln -sfn \"$(pwd)/ray_session\" ~/.rd"
    )
    ctx = ray.init(
        address="local",  # force a fresh local cluster, ignore any existing one
        include_dashboard=True,
        dashboard_host="0.0.0.0",
        dashboard_port=8266,
        num_cpus=NUM_WORKERS + 2,
        _temp_dir=short_temp,
    )
    print("\n" + "=" * 60)
    print(f"Ray dashboard: {ctx.dashboard_url}")
    print("=" * 60 + "\n")

    workers = [Worker.remote(i, SHARDS[i]) for i in range(NUM_WORKERS)]

    w, b, lr = 0.0, 0.0, 0.01
    for step in range(200):
        # Fan out: each worker computes gradient on its shard in parallel.
        grad_refs = [wk.compute_grad.remote(w, b) for wk in workers]
        # Reduce: a separate remote task averages the gradients.
        dw, db = ray.get(average_grads.remote(*grad_refs))
        w -= lr * dw
        b -= lr * db
        if step % 5 == 0:
            loss = sum(((w * x + b) - y) ** 2 for x, y in DATA) / len(DATA)
            print(f"step {step:3d}  w={w:6.3f}  b={b:6.3f}  loss={loss:.4f}")

    print(f"\nDone. final w={w:.4f} b={b:.4f} (target w=2, b=1)")
    print("Dashboard stays up — Ctrl-C to exit.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
