# Simulation code: GenAI semantic video multicast over LEO space data centres

This folder contains the simulator, raw per-run outputs and analysis scripts
behind every number, table and figure in the paper "Generative AI-Assisted
Semantic Video Coding and Onboard Predictive Multicast Adaptation for
Space-Edge Telemedicine".

The simulator is a **time-slotted, flow-level Python simulator**. It is not a
packet-level emulator. Every parameter is a modelling assumption. The single
source of truth is `params.py`, and each raw JSON file also stores the full
parameter set it used.

## Files
| File | Purpose |
|---|---|
| `params.py` | `Params` dataclass with all defaults (constellation, SDCs, ISL slice, traffic, coding, receivers, controllers) |
| `constellation.py` | Walker-delta 53°:576/24/1 at 550 km, circular Keplerian orbits, +Grid ISLs, max-elevation attachment, SDC blocks, border satellites |
| `traffic.py` | Background ISL utilisation: hotspot geography × L + AR(1) + Poisson ramped bursts + per-slot noise |
| `policies.py` | Coding models (conventional SHVC-style / semantic), tree projection, reactive rule, predictive greedy, MILP (PuLP + HiGHS) |
| `sim.py` | One run: geometry, shortest-path multicast trees per 1-s epoch, fluid tail-drop queues per 20-ms slot, all 9 schemes on common random numbers |
| `run_experiments.py` | All configurations × seeds 1–10 → `results/raw/<config>_seed<k>.json` |
| `analyze.py` | Aggregation (mean, 95 % Student-t CI), paired t-tests, figures, LaTeX tables |
| `paired_extra.py` | Extra paired tests quoted in the text → `results/aggregated/paired_extra.txt` |
| `quick.py`, `dbg.py` | Small debugging helpers (not needed to reproduce) |

## Reproduce
```bash
python3 -m venv venv && . venv/bin/activate
pip install -r requirements.txt          # Python 3.13.5 was used
NPROC=5 python run_experiments.py        # 200 runs (20 configs x 10 seeds); 40–120 s per run per core
python analyze.py > results/aggregated/console_summary.txt
python paired_extra.py
```
`run_experiments.py` skips runs whose JSON already exists, so delete
`results/raw/` to rerun from scratch. Seeds are fixed (1–10), and all schemes in a run
share the same geometry, subscriptions and background realisation. MILP
solve times (and therefore `solve_*` fields) depend on the machine. All other
metrics are deterministic for a given seed, apart from possible MILP
tie-breaking differences between HiGHS versions.

## Configurations
* `default`: N=16 receivers, M=16 SDCs (4×4), L=0.7, ρ=4, G_m=3
* `N4, N8, N32, N48`: number of receivers
* `M4, M36, M64`: number of SDCs (2×2, 6×6, 8×8)
* `L0.5, L0.6, L0.8, L0.9`: background load scale
* `rho2, rho6, rho8`: semantic token efficiency (semantic schemes only)
* `dec0.5, dec1.5`: neural decoding latency scale (semantic schemes only)
* `gpu0, gpu1, gpu6`: re-quantisation budget per SDC (semantic schemes only)

## Schemes
UC-H265 (unicast), MC-SVC-S / -R / -PM (conventional layered multicast: static,
reactive, predictive MILP), SEM-S / -R / -PG / -PM (semantic tokens: static,
reactive, predictive greedy, predictive MILP = proposed), SEM-OM (MILP with
oracle next-epoch load; a reference only).

## Metrics (per run, averaged after a 10-s warm-up)
* `isl_mbps`: sum over directed ISLs of the medical rate sent (Mbps·link)
* `ontime_delivery`: mean over (source, receiver) pair-slots of (1 − loss)·1[base latency ≤ 150 ms]
* `lat_base_p95/p99`: percentiles of base-layer latency over pair-slots
* `quality_icu`: quality proxy Q(r) = 100(1 − e^{−r/3}) of the tier received on time, ×(1 − loss), ICU pairs
* `enh_ontime`: share of enhancement-carrying pair-slots whose full latency ≤ 150 ms
* `jitter_ms`, `solve_mean_s`, `solve_max_s`, `n_nonoptimal`, `struct_infeasible`, `mean_hops`, `mean_prop_ms`

## Key modelling assumptions (see `params.py`)
* Medical slice of 100 Mbps per directed ISL, 40-ms tail-drop buffer, fluid queue.
* Conventional cumulative layers 1.5/3.5/5.5/8.0 Mbps. The semantic tier rate is
  r0 + (r_c − r0)/ρ with ρ=4, a fidelity cap of Q=90, and neural decode latency
  10 + {15, 22, 30} ms. Re-quantisation (5 ms) is possible only at SDC border satellites.
* ICU receivers: 25 Mbps downlink, 6 GPU units. Mobile receivers: 2 Mbps, no GPU.
* Holt forecaster α=0.5, β=0.3. MILP η=0.9, λ=0.05, time limit 2 s, gap 0.5 %.
* The quality proxy is not a clinical or perceptual metric. ρ and the decode
  latencies are assumptions, not measured codec values.

## Outputs
* `results/raw/*.json`: 200 per-run outputs (with parameters)
* `results/run_log.txt`: run log with wall times
* `results/aggregated/summary_all.csv`: means and 95 % CIs for every config × scheme × metric
* `results/aggregated/paired_default.txt`, `paired_extra.txt`: paired t-tests
* `results/aggregated/table_*.tex`: tables used in the paper
* `results/figures/*.pdf|png`: figures (the paper uses fig_timeseries, fig_sweep_load, fig_sweep_N)
