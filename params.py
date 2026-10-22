"""All simulation parameters (single source of truth). Values are modelling
assumptions chosen by the authors of the simulation; see README.md."""
from dataclasses import dataclass, field, asdict
import json


@dataclass
class Params:
    # ---------------- constellation (Walker-delta i:T/P/F) ----------------
    planes: int = 24               # P
    sats_per_plane: int = 24       # S  -> T = 576 satellites
    phasing: int = 1               # F
    altitude_km: float = 550.0
    inclination_deg: float = 53.0
    min_elev_deg: float = 25.0     # ground terminal minimum elevation
    # ---------------- SDC clustering ----------------
    clusters_a: int = 4            # clusters along plane index
    clusters_b: int = 4            # clusters along in-plane index -> M = a*b
    gpu_per_cluster: int = 3       # concurrent token re-quantisations per SDC
    # ---------------- ISL / queueing ----------------
    isl_capacity_mbps: float = 100.0  # capacity of the medical slice per directed ISL
    buffer_ms: float = 40.0           # tail-drop buffer (in ms of capacity)
    pkt_bytes: int = 1500
    # ---------------- time ----------------
    sim_s: int = 120
    warmup_s: int = 10
    slot_s: float = 0.02           # fluid-queue slot (20 ms)
    epoch_s: float = 1.0           # control/decision epoch
    # ---------------- background (cross) traffic in the medical slice ----
    load_scale: float = 0.70       # L: mean utilisation scale
    ar_tau_s: float = 10.0
    ar_sigma: float = 0.06
    burst_rate_per_s: float = 1.0 / 300.0   # per directed link
    burst_mag: tuple = (0.20, 0.45)
    burst_ramp_s: float = 3.0
    burst_hold_mean_s: float = 10.0
    fast_sigma: float = 0.05       # per-slot utilisation noise
    # ---------------- traffic sources / receivers ----------------
    n_sources: int = 3
    n_receivers: int = 16
    frac_icu: float = 0.75         # remaining receivers are mobile specialist terminals
    p_subscribe: float = 0.5
    src_box: tuple = (20.0, 30.0, 50.0, 60.0)    # lat_min, lat_max, lon_min, lon_max
    rx_box: tuple = (0.0, 55.0, -10.0, 80.0)
    # ---------------- video / coding model ----------------
    base_mbps: float = 1.5
    conv_cum_mbps: tuple = (1.5, 3.5, 5.5, 8.0)  # SHVC-style cumulative layer rates
    q_r0_mbps: float = 3.0         # Q(R) = 100 (1 - exp(-R/R0))
    sem_gain: float = 4.0          # rho: semantic token efficiency vs. conventional enhancement
    sem_qcap: float = 90.0         # fidelity cap of generative reconstruction
    sem_dec_ms: tuple = (0.0, 15.0, 22.0, 30.0)   # neural decode latency per tier
    sem_dec_units: tuple = (0, 1, 2, 3)           # receiver GPU units per tier
    sem_enc_extra_ms: float = 10.0
    sem_dec_scale: float = 1.0     # multiplies sem_dec_ms (sensitivity study)
    conv_enh_dec_ms: float = 3.0
    requant_ms: float = 5.0        # onboard token re-quantisation latency
    fixed_ms: float = 25.0         # capture+base encode+base decode+packetisation
    deadline_ms: float = 150.0     # clinical interactive bound (ITU-T G.114)
    # ---------------- receivers ----------------
    icu_downlink_mbps: float = 25.0
    icu_gpu_units: int = 6
    mob_downlink_mbps: float = 2.0
    mob_gpu_units: int = 0
    w_icu: float = 1.0
    # ---------------- controllers ----------------
    eta: float = 0.9               # target utilisation in MILP / greedy
    holt_alpha: float = 0.5
    holt_beta: float = 0.3
    milp_time_limit_s: float = 2.0
    milp_gap: float = 0.005
    lambda_bw: float = 0.05
    react_hi_util: float = 0.95
    react_lo_util: float = 0.75
    react_up_epochs: int = 3
    schemes: tuple = ("UC-H265", "MC-SVC-S", "MC-SVC-R", "MC-SVC-PM",
                      "SEM-S", "SEM-R", "SEM-PG", "SEM-PM", "SEM-OM")

    def to_json(self):
        return json.dumps(asdict(self), indent=1)
