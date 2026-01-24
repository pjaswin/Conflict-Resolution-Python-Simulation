"""
RAN sandbox simulator (single-cell) environment module.

This file contains:
- CQI table + traffic model
- UE/config definitions
- PRB->RBG mapping
- Schedulers (RR/PF/WF)
- PHY model (pathloss + shadowing + fading + noise)
- RANSandboxEnv with step() and snapshot() APIs

Note:
- MCS/CQI table is simplified (toy mapping).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

SLICES = ("eMBB", "URLLC", "mMTC")


# -----------------------------
# CQI TABLE (toy but realistic-ish)
# - snr_db: minimum SNR required (dB)
# - eff: spectral efficiency (bits/s/Hz)
# -----------------------------
CQI_TABLE = [
    {"cqi": 1,  "snr_db": -9.478, "eff": 0.15237},
    {"cqi": 2,  "snr_db": -6.658, "eff": 0.23440},
    {"cqi": 3,  "snr_db": -4.098, "eff": 0.37700},
    {"cqi": 4,  "snr_db": -1.798, "eff": 0.60160},
    {"cqi": 5,  "snr_db":  0.399, "eff": 0.87700},
    {"cqi": 6,  "snr_db":  2.424, "eff": 1.17580},
    {"cqi": 7,  "snr_db":  4.489, "eff": 1.47660},
    {"cqi": 8,  "snr_db":  6.367, "eff": 1.91410},
    {"cqi": 9,  "snr_db":  8.456, "eff": 2.40630},
    {"cqi": 10, "snr_db": 10.266, "eff": 2.73050},
    {"cqi": 11, "snr_db": 12.218, "eff": 3.32230},
    {"cqi": 12, "snr_db": 14.122, "eff": 3.90230},
    {"cqi": 13, "snr_db": 15.849, "eff": 4.52340},
    {"cqi": 14, "snr_db": 17.786, "eff": 5.11520},
    {"cqi": 15, "snr_db": 19.809, "eff": 5.55470},
]


# -----------------------------
# Traffic models
# -----------------------------
@dataclass
class TrafficModel:
    kind: str  # "cbr" or "poisson"
    avg_rate_bps: float
    pkt_size_bits: int = 12000  # 1500 bytes * 8

    def arrivals_bits(self, dt_s: float, rng: np.random.Generator) -> int:
        if self.kind == "cbr":
            return int(round(self.avg_rate_bps * dt_s))
        if self.kind == "poisson":
            lam_pkts_per_s = self.avg_rate_bps / float(self.pkt_size_bits)
            n_pkts = int(rng.poisson(lam_pkts_per_s * dt_s))
            return n_pkts * self.pkt_size_bits
        raise ValueError(f"Unknown traffic model kind: {self.kind}")


# -----------------------------
# Helpers
# -----------------------------
def db_to_linear(db: float) -> float:
    return 10 ** (db / 10.0)

def linear_to_db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-30))

def w_to_dbm(p_w: float) -> float:
    return 10.0 * math.log10(max(p_w, 1e-30)) + 30.0

def dbm_to_w(p_dbm: float) -> float:
    return 10 ** ((p_dbm - 30.0) / 10.0)

def noise_power_dbm(bw_hz: float, thermal_noise_dbm_hz: float, noise_figure_db: float) -> float:
    return thermal_noise_dbm_hz + 10.0 * math.log10(bw_hz) + noise_figure_db

def noise_power_w(bw_hz: float, thermal_noise_dbm_hz: float, noise_figure_db: float) -> float:
    return dbm_to_w(noise_power_dbm(bw_hz, thermal_noise_dbm_hz, noise_figure_db))

def cqi_from_snr_db(snr_db: float) -> Tuple[int, float]:
    """
    Returns (cqi_index, spectral_efficiency_bits_per_hz) for the given SNR in dB.
    Picks the highest CQI whose threshold <= snr_db.
    If below CQI1 threshold, returns CQI=0 and eff=0.
    """
    chosen_cqi = 0
    chosen_eff = 0.0
    for entry in CQI_TABLE:
        if snr_db >= entry["snr_db"]:
            chosen_cqi = int(entry["cqi"])
            chosen_eff = float(entry["eff"])
        else:
            break
    return chosen_cqi, chosen_eff


# -----------------------------
# UE / config
# -----------------------------
@dataclass
class UE:
    ue_id: int
    slice_name: str
    distance_m: float
    buffer_bits: int = 0
    avg_thr_bps: float = 1.0  # PF average
    # Slow fading (shadowing) cached per UE
    shadowing_db: float = 0.0
    next_shadow_update_s: float = 0.0

    # For URLLC deadlines
    urllc_pkts: List[Tuple[int, float]] = field(default_factory=list)


@dataclass
class SimConfig:
    # Topology
    n_ues: int = 6
    cell_radius_m: float = 50.0

    # Carrier / propagation
    carrier_freq_ghz: float = 3.5

    # Radio resources
    n_prbs_total: int = 50
    n_rbgs: int = 17
    prb_bw_hz: float = 180e3  # simplified

    # Time
    dt_s: float = 0.01  # 10 ms

    # Buffers
    buffer_max_bits: int = 5_000_000  # per UE

    # Initial slicing (PRBs)
    slice_prbs: Dict[str, int] = field(default_factory=lambda: {"eMBB": 23, "URLLC": 16, "mMTC": 11})

    # Scheduler RR / PF / WF
    scheduler: str = "PF"

    # Transmit power (total)
    total_tx_power_dbm: float = 30.0

    # Noise
    noise_figure_db: float = 13.0
    thermal_noise_dbm_hz: float = -174.0

    # Log-distance pathloss model
    pathloss_exp_n: float = 3.0
    shadowing_std_db: float = 6.0
    shadowing_update_period_s: float = 1.0
    fading: str = "rayleigh"  # "rayleigh" or "none"
    urllc_deadline_s: float = 0.02  # 20 ms

    # PF averaging
    pf_tc_s: float = 1.0

    seed: int = 7


# -----------------------------
# RBG partition
# -----------------------------
def build_rbg_prb_sizes(n_prbs_total: int, n_rbgs: int) -> List[int]:
    base = n_prbs_total // n_rbgs
    rem = n_prbs_total % n_rbgs
    sizes = [base + (1 if i < rem else 0) for i in range(n_rbgs)]
    assert sum(sizes) == n_prbs_total
    return sizes


# -----------------------------
# Schedulers
# -----------------------------
class SchedulerBase:
    name: str
    def pick(self, ues: List[UE], inst_rate_bps: Dict[int, float], rng: np.random.Generator) -> UE:
        raise NotImplementedError

class RoundRobinScheduler(SchedulerBase):
    name = "RR"
    def __init__(self):
        self.ptr: Dict[str, int] = {}

    def pick(self, ues: List[UE], inst_rate_bps: Dict[int, float], rng: np.random.Generator) -> UE:
        s = ues[0].slice_name
        p = self.ptr.get(s, 0) % len(ues)
        ue = ues[p]
        self.ptr[s] = (p + 1) % len(ues)
        return ue

class ProportionalFairScheduler(SchedulerBase):
    name = "PF"
    def pick(self, ues: List[UE], inst_rate_bps: Dict[int, float], rng: np.random.Generator) -> UE:
        best = ues[0]
        best_metric = -1.0
        for ue in ues:
            r = inst_rate_bps[ue.ue_id]
            metric = r / max(ue.avg_thr_bps, 1e-9)
            if metric > best_metric:
                best_metric = metric
                best = ue
        return best

class WaterfillingLikeScheduler(SchedulerBase):
    name = "WF"
    def pick(self, ues: List[UE], inst_rate_bps: Dict[int, float], rng: np.random.Generator) -> UE:
        return max(ues, key=lambda ue: inst_rate_bps[ue.ue_id])

def make_scheduler(name: str) -> SchedulerBase:
    n = name.upper()
    if n == "RR":
        return RoundRobinScheduler()
    if n == "PF":
        return ProportionalFairScheduler()
    if n == "WF":
        return WaterfillingLikeScheduler()
    raise ValueError("scheduler must be one of RR/PF/WF")


# -----------------------------
# PHY aggregation window
# -----------------------------
@dataclass
class PhyAgg:
    ue_rate_sum_bps: Dict[int, float] = field(default_factory=dict)
    ue_snr_sum_lin: Dict[int, float] = field(default_factory=dict)
    ue_noise_sum_w: Dict[int, float] = field(default_factory=dict)
    ue_samples: Dict[int, int] = field(default_factory=dict)
    ue_cqi_hist: Dict[int, Dict[int, int]] = field(default_factory=dict)

    slice_rate_sum_bps: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in SLICES})
    slice_samples: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLICES})

    def init_ues(self, ue_ids: List[int]) -> None:
        for uid in ue_ids:
            self.ue_rate_sum_bps.setdefault(uid, 0.0)
            self.ue_snr_sum_lin.setdefault(uid, 0.0)
            self.ue_noise_sum_w.setdefault(uid, 0.0)
            self.ue_samples.setdefault(uid, 0)
            self.ue_cqi_hist.setdefault(uid, {})

    def add_sample(self, slice_name: str, ue_id: int, rate_bps: float, snr_lin: float, noise_w: float, mcs: int) -> None:
        self.ue_rate_sum_bps[ue_id] += rate_bps
        self.ue_snr_sum_lin[ue_id] += snr_lin
        self.ue_noise_sum_w[ue_id] += noise_w
        self.ue_samples[ue_id] += 1
        self.ue_cqi_hist[ue_id][mcs] = self.ue_cqi_hist[ue_id].get(mcs, 0) + 1

        self.slice_rate_sum_bps[slice_name] += rate_bps
        self.slice_samples[slice_name] += 1


# -----------------------------
# Environment
# -----------------------------
@dataclass
class RANSandboxEnv:
    cfg: SimConfig
    rng: np.random.Generator = field(init=False)

    ues: List[UE] = field(init=False)
    traffic: Dict[str, TrafficModel] = field(init=False)

    rbg_prb_sizes: List[int] = field(init=False)
    rbg_bw_hz: List[float] = field(init=False)

    tx_on: bool = False
    scheduler: SchedulerBase = field(init=False)
    detail_on: bool = True

    urllc_deadline_misses: int = 0
    urllc_generated_pkts: int = 0

    t_s: float = 0.0

    # totals per slice
    total_arrivals_bits: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLICES})
    total_accepted_bits: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLICES})
    total_dropped_bits: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLICES})
    total_sent_bits: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLICES})

    # totals per UE
    ue_arrivals_bits: Dict[int, int] = field(default_factory=dict)
    ue_accepted_bits: Dict[int, int] = field(default_factory=dict)
    ue_dropped_bits: Dict[int, int] = field(default_factory=dict)
    ue_sent_bits: Dict[int, int] = field(default_factory=dict)

    phy_window: PhyAgg = field(default_factory=PhyAgg)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.cfg.seed)

        slice_assign = ["eMBB", "eMBB", "URLLC", "URLLC", "mMTC", "mMTC"]
        if self.cfg.n_ues != 6:
            slice_assign = [SLICES[i % len(SLICES)] for i in range(self.cfg.n_ues)]

        self.ues = []
        for i in range(self.cfg.n_ues):
            d = float(self.rng.uniform(1.0, self.cfg.cell_radius_m))
            ue = UE(ue_id=i, slice_name=slice_assign[i], distance_m=d)
            ue.shadowing_db = float(self.rng.normal(0.0, self.cfg.shadowing_std_db))
            ue.next_shadow_update_s = 0.0
            self.ues.append(ue)

        for ue in self.ues:
            self.ue_arrivals_bits[ue.ue_id] = 0
            self.ue_accepted_bits[ue.ue_id] = 0
            self.ue_dropped_bits[ue.ue_id] = 0
            self.ue_sent_bits[ue.ue_id] = 0

        self.traffic = {
            "eMBB": TrafficModel(kind="cbr", avg_rate_bps=4e6),
            "URLLC": TrafficModel(kind="poisson", avg_rate_bps=89_290.0),
            "mMTC": TrafficModel(kind="poisson", avg_rate_bps=44_640.0),
        }

        self.rbg_prb_sizes = build_rbg_prb_sizes(self.cfg.n_prbs_total, self.cfg.n_rbgs)
        self.rbg_bw_hz = [sz * self.cfg.prb_bw_hz for sz in self.rbg_prb_sizes]

        self.scheduler = make_scheduler(self.cfg.scheduler)
        self.set_slicing(self.cfg.slice_prbs.get("eMBB", 0), self.cfg.slice_prbs.get("URLLC", 0))

        self.phy_window.init_ues([ue.ue_id for ue in self.ues])

        # FSPL @ 1m: 32.4 + 20log10(f_GHz)
        self.pl0_db = 32.4 + 20.0 * math.log10(self.cfg.carrier_freq_ghz)

    def set_slicing(self, embb_prbs: int, urllc_prbs: int) -> None:
        embb_prbs = max(0, min(self.cfg.n_prbs_total, int(embb_prbs)))
        urllc_prbs = max(0, min(self.cfg.n_prbs_total, int(urllc_prbs)))
        tot = embb_prbs + urllc_prbs
        if tot > self.cfg.n_prbs_total:
            scale = self.cfg.n_prbs_total / tot
            embb_prbs = int(math.floor(embb_prbs * scale))
            urllc_prbs = int(math.floor(urllc_prbs * scale))
        mmtc_prbs = self.cfg.n_prbs_total - (embb_prbs + urllc_prbs)
        self.cfg.slice_prbs = {"eMBB": embb_prbs, "URLLC": urllc_prbs, "mMTC": mmtc_prbs}

    def _prbs_budget_to_rbg_indices(self, prbs_budget: int, forbidden: set[int]) -> List[int]:
        chosen = []
        remaining = prbs_budget
        for i in range(self.cfg.n_rbgs):
            if i in forbidden:
                continue
            if remaining <= 0:
                break
            sz = self.rbg_prb_sizes[i]
            if sz <= remaining:
                chosen.append(i)
                remaining -= sz
        return chosen

    def rbg_allocation_by_slice(self) -> Dict[str, List[int]]:
        used = set()
        embb_rbgs = self._prbs_budget_to_rbg_indices(self.cfg.slice_prbs["eMBB"], used)
        used |= set(embb_rbgs)
        urllc_rbgs = self._prbs_budget_to_rbg_indices(self.cfg.slice_prbs["URLLC"], used)
        used |= set(urllc_rbgs)
        mmtc_rbgs = [i for i in range(self.cfg.n_rbgs) if i not in used]
        return {"eMBB": embb_rbgs, "URLLC": urllc_rbgs, "mMTC": mmtc_rbgs}

    def pathloss_db(self, ue: UE) -> float:
        d = max(ue.distance_m, 1.0)

        # Slow shadowing update (not every TTI)
        if self.t_s >= ue.next_shadow_update_s:
            ue.shadowing_db = float(self.rng.normal(0.0, self.cfg.shadowing_std_db))
            ue.next_shadow_update_s = self.t_s + self.cfg.shadowing_update_period_s

        return self.pl0_db + 10.0 * self.cfg.pathloss_exp_n * math.log10(d) + ue.shadowing_db

    def fading_power_linear(self) -> float:
        if self.cfg.fading == "none":
            return 1.0
        if self.cfg.fading == "rayleigh":
            return float(self.rng.exponential(1.0))  # |h|^2
        raise ValueError("fading must be 'rayleigh' or 'none'")

    def inst_phy(self, ue: UE, bw_hz: float, p_rbg_w: float) -> Tuple[float, float, float, int, float]:
        n_w = noise_power_w(bw_hz, self.cfg.thermal_noise_dbm_hz, self.cfg.noise_figure_db)
        n_dbm = noise_power_dbm(bw_hz, self.cfg.thermal_noise_dbm_hz, self.cfg.noise_figure_db)

        p_tx_dbm = w_to_dbm(p_rbg_w)

        pl_db = self.pathloss_db(ue)
        fad_lin = self.fading_power_linear()
        fad_db = linear_to_db(fad_lin)

        p_rx_dbm = p_tx_dbm - pl_db + fad_db
        snr_db = p_rx_dbm - n_dbm
        snr_lin = db_to_linear(snr_db)

        cqi, eff = cqi_from_snr_db(snr_db)
        rate_bps = bw_hz * eff
        return rate_bps, snr_lin, n_w, cqi, eff

    def step(self) -> None:
        dt = self.cfg.dt_s

        # Arrivals
        for ue in self.ues:
            bits_in = self.traffic[ue.slice_name].arrivals_bits(dt, self.rng)

            # URLLC: generate packets explicitly with deadlines
            if ue.slice_name == "URLLC":
                tm = self.traffic["URLLC"]
                lam_pkts_per_s = tm.avg_rate_bps / float(tm.pkt_size_bits)
                n_pkts = int(self.rng.poisson(lam_pkts_per_s * dt))
                bits_in = n_pkts * tm.pkt_size_bits

                for _ in range(n_pkts):
                    self.urllc_generated_pkts += 1
                    ue.urllc_pkts.append((tm.pkt_size_bits, self.t_s + self.cfg.urllc_deadline_s))

            self.total_arrivals_bits[ue.slice_name] += bits_in
            self.ue_arrivals_bits[ue.ue_id] += bits_in

            room = self.cfg.buffer_max_bits - ue.buffer_bits

            if ue.slice_name != "URLLC":
                accepted = max(0, min(bits_in, room))
                dropped = max(0, bits_in - accepted)
                ue.buffer_bits += accepted
            else:
                # Accept URLLC packets atomically; enforce buffer cap with tail drop
                dropped = 0
                while ue.buffer_bits + sum(b for b, _dl in ue.urllc_pkts) > self.cfg.buffer_max_bits:
                    b, _dl = ue.urllc_pkts.pop()
                    dropped += b
                accepted = bits_in - dropped
                ue.buffer_bits += accepted

            self.total_accepted_bits[ue.slice_name] += accepted
            self.total_dropped_bits[ue.slice_name] += dropped
            self.ue_accepted_bits[ue.ue_id] += accepted
            self.ue_dropped_bits[ue.ue_id] += dropped

        # --- URLLC deadline expiration (drop overdue packets) ---
        for ue in self.ues:
            if ue.slice_name != "URLLC" or not ue.urllc_pkts:
                continue
            kept = []
            for (b, dl) in ue.urllc_pkts:
                if self.t_s > dl:
                    self.urllc_deadline_misses += 1
                    ue.buffer_bits -= b

                    self.total_dropped_bits["URLLC"] += b
                    self.ue_dropped_bits[ue.ue_id] += b
                else:
                    kept.append((b, dl))
            ue.urllc_pkts = kept

        # Transmission
        if self.tx_on:
            slice_rbgs = self.rbg_allocation_by_slice()

            # Group UEs by slice (needed for "active" bandwidth)
            ues_by_slice: Dict[str, List[UE]] = {s: [] for s in SLICES}
            for ue in self.ues:
                ues_by_slice[ue.slice_name].append(ue)

            # --- Constant PSD: allocate power proportional to RBG bandwidth ---
            p_total_w = dbm_to_w(self.cfg.total_tx_power_dbm)

            used_rbg_indices: List[int] = []
            for s in SLICES:
                if len(ues_by_slice[s]) > 0:
                    used_rbg_indices.extend(slice_rbgs[s])

            total_active_bw_hz = sum(self.rbg_bw_hz[i] for i in used_rbg_indices)
            if total_active_bw_hz <= 0:
                total_active_bw_hz = 1.0  # safety

            sent_bits_by_ue: Dict[int, int] = {ue.ue_id: 0 for ue in self.ues}
            sent_bits_by_slice: Dict[str, int] = {s: 0 for s in SLICES}

            self.phy_window.init_ues([ue.ue_id for ue in self.ues])

            for s in SLICES:
                rbgs = slice_rbgs[s]
                ues_s = ues_by_slice[s]
                if not rbgs or not ues_s:
                    continue

                for rbg_i in rbgs:
                    bw = self.rbg_bw_hz[rbg_i]

                    inst_rate: Dict[int, float] = {}
                    inst_snr: Dict[int, float] = {}
                    inst_noise: Dict[int, float] = {}
                    inst_mcs: Dict[int, int] = {}

                    for ue in ues_s:
                        p_rbg_w = p_total_w * (bw / total_active_bw_hz)
                        rate, snr_lin, n_w, mcs, _eff = self.inst_phy(ue, bw, p_rbg_w)
                        inst_rate[ue.ue_id] = rate
                        inst_snr[ue.ue_id] = snr_lin
                        inst_noise[ue.ue_id] = n_w
                        inst_mcs[ue.ue_id] = mcs

                    chosen = self.scheduler.pick(ues_s, inst_rate, self.rng)

                    self.phy_window.add_sample(
                        slice_name=s,
                        ue_id=chosen.ue_id,
                        rate_bps=inst_rate[chosen.ue_id],
                        snr_lin=inst_snr[chosen.ue_id],
                        noise_w=inst_noise[chosen.ue_id],
                        mcs=inst_mcs[chosen.ue_id],
                    )

                    cap_bits = int(math.floor(inst_rate[chosen.ue_id] * dt))

                    if chosen.slice_name != "URLLC":
                        send_bits = min(chosen.buffer_bits, cap_bits)
                        chosen.buffer_bits -= send_bits
                    else:
                        # Serve URLLC packets FIFO
                        remaining = cap_bits
                        sent = 0
                        new_q = []
                        for (b, dl) in chosen.urllc_pkts:
                            if remaining <= 0:
                                new_q.append((b, dl))
                                continue
                            take = min(b, remaining)
                            b2 = b - take
                            remaining -= take
                            sent += take
                            if b2 > 0:
                                new_q.append((b2, dl))
                        chosen.urllc_pkts = new_q
                        chosen.buffer_bits -= sent
                        send_bits = sent

                    sent_bits_by_ue[chosen.ue_id] += send_bits
                    sent_bits_by_slice[s] += send_bits

            for s in SLICES:
                self.total_sent_bits[s] += sent_bits_by_slice[s]
            for ue_id, b in sent_bits_by_ue.items():
                self.ue_sent_bits[ue_id] += b

            alpha = max(0.0, min(1.0, dt / max(self.cfg.pf_tc_s, 1e-9)))
            for ue in self.ues:
                inst_thr = sent_bits_by_ue[ue.ue_id] / dt
                ue.avg_thr_bps = (1 - alpha) * ue.avg_thr_bps + alpha * inst_thr

        self.t_s += dt

    def snapshot(self) -> Dict:
        per_slice_buffer = {s: 0 for s in SLICES}
        for ue in self.ues:
            per_slice_buffer[ue.slice_name] += ue.buffer_bits
        return {
            "t_s": self.t_s,
            "tx_on": self.tx_on,
            "scheduler": self.scheduler.name,
            "slice_prbs": dict(self.cfg.slice_prbs),
            "slice_buffer_bits": per_slice_buffer,
            "total_arrivals_bits": dict(self.total_arrivals_bits),
            "total_accepted_bits": dict(self.total_accepted_bits),
            "total_dropped_bits": dict(self.total_dropped_bits),
            "total_sent_bits": dict(self.total_sent_bits),
        }

    def ue_snapshot(self) -> List[Dict]:
        return [
            {
                "ue": ue.ue_id,
                "slice": ue.slice_name,
                "dist_m": round(ue.distance_m, 2),
                "buffer_bits": ue.buffer_bits,
                "avg_thr_bps_pf": ue.avg_thr_bps,
                "arrivals_bits": self.ue_arrivals_bits[ue.ue_id],
                "accepted_bits": self.ue_accepted_bits[ue.ue_id],
                "dropped_bits": self.ue_dropped_bits[ue.ue_id],
                "sent_bits": self.ue_sent_bits[ue.ue_id],
            }
            for ue in self.ues
        ]

    def pop_phy_window(self) -> PhyAgg:
        out = self.phy_window
        self.phy_window = PhyAgg()
        self.phy_window.init_ues([ue.ue_id for ue in self.ues])
        return out
