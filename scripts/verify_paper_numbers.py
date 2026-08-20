# -*- coding: utf-8 -*-
"""
Reproduce Table I of the paper and the 5.1% headline number.

Loads the five trained checkpoints and the offline dataset, replays every row of
Table I, and prints the latency improvement of CQL+Mask over the rule-based
baseline. No GPU, no retraining, no new handshake collection. Takes a few seconds.

Run from the repository root:
    python scripts/verify_paper_numbers.py

Expected tail of the output:
    CQL+Mask 500.9 ms vs Rule 527.7 ms
    HEADLINE: 5.1% lower latency, 0.0% violations

The forward passes below are reimplemented directly from the saved state_dicts
rather than imported from hybrid_pqc_tls.rl_models, so that verification does not
depend on the training code or on optional packages.
"""

import sys
import importlib.util as iu

import numpy as np
import torch

RULE_VARIANT = "published"  # matches RuleBasedPolicy in hybrid_pqc_tls/rl_evaluate_v2.py

RTT_TOL = 5  # matches run_action_masking_eval.py, which produced Table I

# hybrid_pqc_tls/__init__.py imports stable_baselines3, so load rl_config directly
_spec = iu.spec_from_file_location("rlcfg", "hybrid_pqc_tls/rl_config.py")
rlcfg = iu.module_from_spec(_spec)
sys.modules["rlcfg"] = rlcfg
_spec.loader.exec_module(rlcfg)

ACTION_LIST = rlcfg.ACTION_LIST
ACTION_TO_IDX = rlcfg.ACTION_TO_IDX
NUM_ACTIONS = rlcfg.NUM_ACTIONS
SAFE = np.array([rlcfg.is_pqc_safe_action(i) for i in range(NUM_ACTIONS)])

D = np.load("results/rl/offline_rl_dataset_v2.npz", allow_pickle=True)
A, R, RTT, LAT, WIRE = D["A"], D["R"], D["rtt"], D["latency"], D["wire"]
TEST = D["test_idx"]
RTT_TEST = RTT[TEST]
S_TEST = torch.FloatTensor(D["S"])[TEST]

_cache = {}


def outcome(action, rtt):
    """Mean measured latency, wire size and reward for this action at this RTT."""
    key = (int(action), float(rtt))
    if key not in _cache:
        m = (A == action) & (np.abs(RTT - rtt) < RTT_TOL)
        if m.sum() == 0:
            m = A == action
        _cache[key] = (LAT[m].mean(), WIRE[m].mean(), R[m].mean())
    return _cache[key]


def summarize(actions):
    lat = np.array([outcome(a, r)[0] for a, r in zip(actions, RTT_TEST)])
    wir = np.array([outcome(a, r)[1] for a, r in zip(actions, RTT_TEST)])
    rew = np.array([outcome(a, r)[2] for a, r in zip(actions, RTT_TEST)])
    pol = [ACTION_LIST[a][0] for a in actions]
    lev = [ACTION_LIST[a][1] for a in actions]
    return dict(
        reward=rew.mean(), latency=lat.mean(), wire=wir.mean(),
        hybrid=100 * np.mean([p == "REQUIRE_HYBRID" for p in pol]),
        pqc=100 * np.mean([p == "PQC_ONLY" for p in pol]),
        classical=100 * np.mean([p == "CLASSICAL_ONLY" for p in pol]),
        violation=100 * np.mean([l < 3 for l in lev]),
    )


# --------------------------------------------------------------------- baselines
def rule_policy(rtt):
    """Rule-based baseline, identical to RuleBasedPolicy in rl_evaluate_v2.py."""
    if rtt < 50:
        return ACTION_TO_IDX[("REQUIRE_HYBRID", 5)]
    if rtt < 100:
        return ACTION_TO_IDX[("REQUIRE_HYBRID", 3)]
    if rtt < 150:
        return ACTION_TO_IDX[("PQC_ONLY", 3)]
    return ACTION_TO_IDX[("ALLOW_FALLBACK", 3)]


def oracle_policy(rtt):
    return ACTION_TO_IDX[("REQUIRE_HYBRID", 3)]


# --------------------------------------------------------------------- RL policies
def _mlp(sd, layer_idx):
    """Forward pass for the shared 15-256-256-12 MLP. Dropout is identity at eval."""
    def fwd(x):
        for i, j in enumerate(layer_idx):
            x = x @ sd[f"net.net.{j}.weight"].T + sd[f"net.net.{j}.bias"]
            if i < len(layer_idx) - 1:
                x = torch.relu(x)
        return x
    return fwd


def _ck(name):
    return torch.load(f"results/rl/models/{name}_model.pt",
                      map_location="cpu", weights_only=False)


def rl_scores():
    """Per-method score matrix over the 12 actions. argmax gives the unmasked action."""
    out = {}
    out["CQL"] = _mlp(_ck("cql")["q_net"], [0, 3, 6])(S_TEST)
    out["BC"] = _mlp(_ck("bc")["policy"], [0, 3, 6])(S_TEST)
    out["IQL"] = _mlp(_ck("iql")["policy"], [0, 2, 4])(S_TEST)
    out["AWAC"] = _mlp(_ck("awac")["policy"], [0, 2, 4])(S_TEST)

    b = _ck("bcq")
    probs = torch.softmax(_mlp(b["bc_policy"], [0, 2, 4])(S_TEST), dim=-1)
    keep = (probs >= 0.3 * probs.max(dim=-1, keepdim=True).values).float()
    out["BCQ"] = _mlp(b["q_net"], [0, 2, 4])(S_TEST) - 1e8 * (1 - keep)
    return out


def apply_mask(scores):
    """Inference-time action masking: unsafe actions get -inf, so are never selected."""
    masked = scores.clone()
    masked[:, ~torch.tensor(SAFE)] = float("-inf")
    return masked


# --------------------------------------------------------------------- main
def main():
    hdr = f"{'Method':<14}{'Reward':>8}{'Latency':>9}{'Wire':>7}{'Hybrid':>8}{'PQC':>7}{'Class':>7}{'Viol':>7}"
    print(f"Rule variant: {RULE_VARIANT}\n")
    print(hdr)
    print("-" * len(hdr))

    rows_written = []

    def row(name, actions):
        m = summarize(actions)
        rows_written.append((name, m))
        print(f"{name:<14}{m['reward']:>8.2f}{m['latency']:>9.1f}{m['wire']:>7.0f}"
              f"{m['hybrid']:>8.1f}{m['pqc']:>7.1f}{m['classical']:>7.1f}{m['violation']:>7.1f}")
        return m

    oracle = row("Oracle", [oracle_policy(r) for r in RTT_TEST])
    rule = row("Rule", [rule_policy(r) for r in RTT_TEST])

    scores = rl_scores()
    order = ["CQL", "BC", "IQL", "BCQ", "AWAC"]
    print()
    for n in order:
        row(n, scores[n].argmax(-1).numpy())
    print()
    masked = {}
    for n in order:
        masked[n] = row(n + "+Mask", apply_mask(scores[n]).argmax(-1).numpy())

    # persist the reproduced table so results/ never drifts from the code
    import csv, os
    out = "results/rl/evaluation/table_i.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Method", "Reward", "Latency_ms", "Wire_B",
                    "Hybrid_pct", "PQC_pct", "Classical_pct", "Violation_pct"])
        for name, m in rows_written:
            w.writerow([name, f"{m['reward']:.2f}", f"{m['latency']:.1f}",
                        f"{m['wire']:.0f}", f"{m['hybrid']:.1f}", f"{m['pqc']:.1f}",
                        f"{m['classical']:.1f}", f"{m['violation']:.1f}"])
    print(f"\nwrote {out}")

    cqlm = masked["CQL"]
    imp = 100 * (rule["latency"] - cqlm["latency"]) / rule["latency"]
    print()
    print(f"CQL+Mask {cqlm['latency']:.1f} ms vs Rule {rule['latency']:.1f} ms")
    print(f"HEADLINE: {imp:.1f}% lower latency, {cqlm['violation']:.1f}% violations")
    print(f"Masking cost: {cqlm['latency'] - summarize(scores['CQL'].argmax(-1).numpy())['latency']:.1f} ms")
    print(f"Oracle gap:   {cqlm['latency'] - oracle['latency']:.1f} ms")


if __name__ == "__main__":
    main()
