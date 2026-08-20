# Dataset description

## results/eval_grid/handshake_raw.csv

12,000 rows: 5 RTT conditions x 12 actions x 200 trials.

Each row is one complete handshake executed against the local testbed with real
ML-KEM and ML-DSA operations. `using_mock_pqc` is `False` on every row.

Selected columns:

| Column | Meaning |
|---|---|
| `policy` | REQUIRE_HYBRID, PQC_ONLY, ALLOW_FALLBACK, CLASSICAL_ONLY |
| `level_int` | NIST security level 1, 3, or 5 |
| `rtt_ms` | emulated one-way delay setting: 0, 15, 50, 100, 200 |
| `total_time_ms` | wall-clock handshake duration |
| `wire_srv_ecdh`, `wire_cli_ecdh` | classical key share bytes |
| `wire_srv_pqc`, `wire_cli_pqc` | post-quantum key share and signature bytes |
| `using_mock_pqc` | always False; real primitives throughout |

Timing is wall-clock from a monotonic counter around real cryptographic work.
Network delay is emulated per handshake flight, not measured over a live link.

## results/rl/offline_rl_dataset_v2.npz

10,000 transitions sampled from the grid by epsilon-greedy behavioral resampling
(epsilon = 0.3) around the per-RTT reward-optimal action, split 8,000 train and
2,000 test.

| Array | Shape | Meaning |
|---|---|---|
| `S` | (10000, 15) | normalized state features |
| `A` | (10000,) | action index, 0 to 11 |
| `R` | (10000,) | labeled reward |
| `rtt` | (10000,) | RTT for the transition |
| `latency` | (10000,) | measured handshake latency, ms |
| `wire` | (10000,) | total wire bytes |
| `train_idx`, `test_idx` | index arrays | the split used in the paper |

The near-deterministic behavior policy is why the four non-conservative
algorithms collapse to a single policy. See the README section on known issues.

## results/rl/models/*.pt

Five checkpoints: `bc`, `cql`, `iql`, `bcq`, `awac`. Each is a dict of
state_dicts. Layer indices differ between architectures; see
`scripts/verify_paper_numbers.py` for the exact forward passes used to reproduce
the published numbers without importing the training code.
