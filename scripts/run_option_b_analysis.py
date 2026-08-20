# -*- coding: utf-8 -*-
"""
Supplementary analyses reported in the paper.

Produces three artifacts from the committed dataset and checkpoints. No
retraining, no new handshakes. Takes about ten seconds.

  1. Random-Safe control row      separates the contribution of the learned
                                  policy from that of action masking
  2. Policy agreement statistic   shows BC/IQL/BCQ/AWAC converge to one policy
  3. Reward weight sensitivity    closed-form stability ranges for each weight

Run from the repository root:
    python scripts/run_option_b_analysis.py

Outputs to results/rl/evaluation/option_b/:
    decomposition.csv, agreement.txt, reward_sensitivity.csv
"""

import os
import sys
import itertools
import importlib.util as iu

import numpy as np
import pandas as pd
import torch

OUT = "results/rl/evaluation/option_b"
os.makedirs(OUT, exist_ok=True)

# hybrid_pqc_tls/__init__.py pulls in stable_baselines3, so load rl_config directly
_spec = iu.spec_from_file_location("rlcfg", "hybrid_pqc_tls/rl_config.py")
rlcfg = iu.module_from_spec(_spec)
sys.modules["rlcfg"] = rlcfg
_spec.loader.exec_module(rlcfg)

ACTION_LIST = rlcfg.ACTION_LIST
ACTION_TO_IDX = rlcfg.ACTION_TO_IDX
NUM_ACTIONS = rlcfg.NUM_ACTIONS
is_safe = rlcfg.is_pqc_safe_action
CFG = rlcfg.DEFAULT_REWARD_CONFIG

DATA = np.load("results/rl/offline_rl_dataset_v2.npz", allow_pickle=True)
A, R, RTT, LAT, WIRE = DATA["A"], DATA["R"], DATA["rtt"], DATA["latency"], DATA["wire"]
TEST = DATA["test_idx"]
RTT_TEST = RTT[TEST]

# RTT tolerance of 5 matches run_action_masking_eval.py, which produced Table I
_cache = {}


def outcome(action, rtt, tol=5):
    key = (int(action), float(rtt))
    if key not in _cache:
        m = (A == action) & (np.abs(RTT - rtt) < tol)
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
        reward=rew.mean(),
        latency=lat.mean(),
        wire=wir.mean(),
        hybrid=100 * np.mean([p == "REQUIRE_HYBRID" for p in pol]),
        pqc=100 * np.mean([p == "PQC_ONLY" for p in pol]),
        classical=100 * np.mean([p == "CLASSICAL_ONLY" for p in pol]),
        violation=100 * np.mean([l < 3 for l in lev]),
    )


# ---------------------------------------------------------------- 1. decomposition
def rule_policy(rtt):
    """Rule-based baseline, identical to RuleBasedPolicy in rl_evaluate_v2.py."""
    if rtt < 50:
        return ACTION_TO_IDX[("REQUIRE_HYBRID", 5)]
    if rtt < 100:
        return ACTION_TO_IDX[("REQUIRE_HYBRID", 3)]
    if rtt < 150:
        return ACTION_TO_IDX[("PQC_ONLY", 3)]
    return ACTION_TO_IDX[("ALLOW_FALLBACK", 3)]


def decomposition(n_seeds=200):
    safe = [i for i in range(NUM_ACTIONS) if is_safe(i)]
    rows = []

    rows.append(dict(Method="Oracle", **summarize(
        [ACTION_TO_IDX[("REQUIRE_HYBRID", 3)]] * len(TEST))))
    rows.append(dict(Method="Rule", **summarize(
        [rule_policy(r) for r in RTT_TEST])))

    draws = []
    for seed in range(n_seeds):
        rg = np.random.default_rng(seed)
        draws.append(summarize(rg.choice(safe, size=len(TEST))))
    rs = {k: np.mean([d[k] for d in draws]) for k in draws[0]}
    rs_sd = {k: np.std([d[k] for d in draws]) for k in draws[0]}
    rows.append(dict(Method="Random-Safe", **rs))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/decomposition.csv", index=False)
    print("\n=== 1. DECOMPOSITION (new Random-Safe control) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"    Random-Safe latency sd over {n_seeds} seeds: {rs_sd['latency']:.2f} ms")
    print(f"    n safe actions after masking: {len(safe)}")
    return df


# ---------------------------------------------------------------- 2. agreement
def _mlp(sd, layer_idx):
    def fwd(x):
        for i, j in enumerate(layer_idx):
            x = x @ sd[f"net.net.{j}.weight"].T + sd[f"net.net.{j}.bias"]
            if i < len(layer_idx) - 1:
                x = torch.relu(x)
        return x
    return fwd


def agreement():
    S = torch.FloatTensor(DATA["S"])[TEST]

    def ck(name):
        return torch.load(f"results/rl/models/{name}_model.pt",
                          map_location="cpu", weights_only=False)

    acts = {}
    acts["BC"] = _mlp(ck("bc")["policy"], [0, 3, 6])(S).argmax(-1).numpy()
    acts["CQL"] = _mlp(ck("cql")["q_net"], [0, 3, 6])(S).argmax(-1).numpy()
    acts["IQL"] = _mlp(ck("iql")["policy"], [0, 2, 4])(S).argmax(-1).numpy()
    acts["AWAC"] = _mlp(ck("awac")["policy"], [0, 2, 4])(S).argmax(-1).numpy()

    b = ck("bcq")
    probs = torch.softmax(_mlp(b["bc_policy"], [0, 2, 4])(S), dim=-1)
    keep = (probs >= 0.3 * probs.max(dim=-1, keepdim=True).values).float()
    q = _mlp(b["q_net"], [0, 2, 4])(S)
    acts["BCQ"] = (q - 1e8 * (1 - keep)).argmax(-1).numpy()

    four = ["BC", "IQL", "BCQ", "AWAC"]
    lines = []
    for a, b_ in itertools.combinations(four, 2):
        lines.append(f"{a} vs {b_}: {100 * (acts[a] == acts[b_]).mean():.2f}%")
    same = np.ones(len(TEST), bool)
    for f in four[1:]:
        same &= acts[f] == acts["BC"]
    lines.append(f"ALL FOUR identical on {100 * same.mean():.2f}% of {len(TEST)} test samples")
    lines.append(f"CQL vs the consensus: {100 * (acts['CQL'] == acts['BC']).mean():.2f}%")

    diff = acts["CQL"] != acts["BC"]
    lines.append(f"CQL diverges on {diff.sum()} samples")
    from collections import Counter
    lines.append("  consensus picks there: " + str(Counter(
        f"{ACTION_LIST[a][0]}_L{ACTION_LIST[a][1]}" for a in acts["BC"][diff]).most_common()))
    lines.append("  CQL picks there:       " + str(Counter(
        f"{ACTION_LIST[a][0]}_L{ACTION_LIST[a][1]}" for a in acts["CQL"][diff]).most_common()))

    txt = "\n".join(lines)
    open(f"{OUT}/agreement.txt", "w").write(txt)
    print("\n=== 2. POLICY AGREEMENT ===")
    print(txt)
    return acts


# ---------------------------------------------------------------- 3. sensitivity
def _grid():
    df = pd.read_csv("results/eval_grid/handshake_raw.csv")
    df["wire_kb"] = (df.wire_srv_ecdh + df.wire_cli_ecdh
                     + df.wire_srv_pqc + df.wire_cli_pqc) / 1000.0
    g = (df.groupby(["policy", "level_int", "rtt_ms"])
           .agg(lat=("total_time_ms", "mean"), wkb=("wire_kb", "mean")).reset_index())
    return {(r.policy, int(r.level_int), float(r.rtt_ms)): (r.lat, r.wkb)
            for r in g.itertuples()}


def sensitivity():
    tab = _grid()
    rtts = [0.0, 15.0, 50.0, 100.0, 200.0]

    def reward(pol, lev, rtt, alpha_base, hyb, beta, gamma):
        lat, wkb = tab[(pol, lev, rtt)]
        alpha = alpha_base + CFG.alpha_rtt_scale * rtt
        if rtt <= CFG.low_rtt_threshold:
            sw = 1.5 * gamma
        elif rtt >= CFG.high_rtt_threshold:
            sw = 0.8 * gamma
        else:
            t = ((rtt - CFG.low_rtt_threshold)
                 / (CFG.high_rtt_threshold - CFG.low_rtt_threshold))
            sw = gamma * (1.5 - 0.7 * t)
        mb = {"REQUIRE_HYBRID": hyb, "PQC_ONLY": CFG.pqc_bonus,
              "ALLOW_FALLBACK": -CFG.fallback_penalty,
              "CLASSICAL_ONLY": -CFG.classical_penalty}[pol]
        viol = CFG.level_violation_penalty if lev < CFG.min_acceptable_level else 0.0
        return 1.0 - alpha * lat - beta * wkb + sw * (lev / 5.0) + mb - viol

    def argmax(**kw):
        return [max(ACTION_LIST, key=lambda a: reward(a[0], a[1], r, **kw)) for r in rtts]

    base = dict(alpha_base=CFG.alpha_base, hyb=CFG.hybrid_bonus,
                beta=CFG.beta, gamma=CFG.gamma)
    rows = []
    sweeps = {
        "alpha_base": [0.0, 0.002, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1],
        "hyb":        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
        "beta":       [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
        "gamma":      [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
    }
    for key, vals in sweeps.items():
        for v in vals:
            kw = dict(base)
            kw[key] = v
            sel = argmax(**kw)
            rows.append(dict(
                weight=key, value=v,
                argmax_per_rtt=" | ".join(f"{p}_L{l}" for p, l in sel),
                unchanged_from_default=(sel == argmax(**base))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/reward_sensitivity.csv", index=False)
    print("\n=== 3. REWARD WEIGHT SENSITIVITY (analytical, no retraining) ===")
    print(f"default weights -> {' | '.join(f'{p}_L{l}' for p, l in argmax(**base))}")
    for key in sweeps:
        sub = df[df.weight == key]
        ok = sub[sub.unchanged_from_default].value
        print(f"  {key:11s} default {base[key]:<7g} "
              f"stable over [{ok.min():g}, {ok.max():g}]")
    return df


if __name__ == "__main__":
    decomposition()
    agreement()
    sensitivity()
    print(f"\nwrote artifacts to {OUT}/")
