"""
this code is based on GPT https://chatgpt.com/c/696419ed-8b5c-832d-b875-3b5a87ebf49a. Changed when Shannon capacity is replaced with lookup table.
"""

# RAN sandbox simulator with:
# - Traffic (CBR + Poisson) per slice
# - Per-UE buffers with accepted/dropped
# - PRB->RBG mapping (50 PRBs -> 17 RBGs)
# - Scheduling RR/PF/WF
# - PHY: realistic pathloss calibration using log-distance model with FSPL(1m) @ 3.5 GHz
# - Shadowing (dB) + Rayleigh fading (power)
# - Noise power from -174 dBm/Hz + NF
# - SNR (single cell) + Shannon capacity
# - Rich per-second outputs: per-slice and per-UE (SNR, noise, avg RBG rate)
#
# Controls:
#   [Enter]                 toggle TX ON/OFF
#   rr | pf | wf            set scheduler
#   slice <e> <u>           set PRBs for eMBB and URLLC (mMTC gets remainder)
#   detail on|off           per-UE PHY details each second
#   status                  print current config
#   q                       quit

from __future__ import annotations

import math
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

SLICES = ("eMBB", "URLLC", "mMTC")


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
    # N(dBm) = -174 + 10log10(BW) + NF
    return thermal_noise_dbm_hz + 10.0 * math.log10(bw_hz) + noise_figure_db

def noise_power_w(bw_hz: float, thermal_noise_dbm_hz: float, noise_figure_db: float) -> float:
    return dbm_to_w(noise_power_dbm(bw_hz, thermal_noise_dbm_hz, noise_figure_db))


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


@dataclass
class SimConfig:
    # Topology
    n_ues: int = 6
    cell_radius_m: float = 50.0

    # Carrier / propagation (common scenario)
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

    # Transmit power (small-cell-ish): 30 dBm total = 1 W
    total_tx_power_dbm: float = 30.0

    # Noise
    noise_figure_db: float = 7.0
    thermal_noise_dbm_hz: float = -174.0

    # Log-distance pathloss model
    pathloss_exp_n: float = 3.0          # typical urban-ish
    shadowing_std_db: float = 6.0        # typical
    fading: str = "rayleigh"             # "rayleigh" or "none"

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

    slice_rate_sum_bps: Dict[str, float] = field(default_factory=lambda: {s: 0.0 for s in SLICES})
    slice_samples: Dict[str, int] = field(default_factory=lambda: {s: 0 for s in SLICES})

    def init_ues(self, ue_ids: List[int]) -> None:
        for uid in ue_ids:
            self.ue_rate_sum_bps.setdefault(uid, 0.0)
            self.ue_snr_sum_lin.setdefault(uid, 0.0)
            self.ue_noise_sum_w.setdefault(uid, 0.0)
            self.ue_samples.setdefault(uid, 0)

    def add_sample(self, slice_name: str, ue_id: int, rate_bps: float, snr_lin: float, noise_w: float) -> None:
        self.ue_rate_sum_bps[ue_id] += rate_bps
        self.ue_snr_sum_lin[ue_id] += snr_lin
        self.ue_noise_sum_w[ue_id] += noise_w
        self.ue_samples[ue_id] += 1

        self.slice_rate_sum_bps[slice_name] += rate_bps
        self.slice_samples[slice_name] += 1


# -----------------------------
# Environment with realistic pathloss
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

        # UE slice membership: 2 UEs per slice by default
        slice_assign = ["eMBB", "eMBB", "URLLC", "URLLC", "mMTC", "mMTC"]
        if self.cfg.n_ues != 6:
            slice_assign = [SLICES[i % len(SLICES)] for i in range(self.cfg.n_ues)]

        self.ues = []
        for i in range(self.cfg.n_ues):
            # NOTE: uniform in distance (simple). For uniform in area, draw sqrt(U)*R.
            d = float(self.rng.uniform(1.0, self.cfg.cell_radius_m))
            self.ues.append(UE(ue_id=i, slice_name=slice_assign[i], distance_m=d))

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

        # Precompute FSPL at 1 meter (dB) for given carrier freq:
        # PL0(dB) ≈ 32.4 + 20 log10(f_GHz)
        self.pl0_db = 32.4 + 20.0 * math.log10(self.cfg.carrier_freq_ghz)

    # ---------- slicing ----------
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

    # ---------- realistic pathloss + fading ----------
    def pathloss_db(self, d_m: float) -> float:
        """
        Log-distance model:
          PL(dB) = PL0(dB @ 1m) + 10*n*log10(d) + shadowing
        where:
          PL0 ≈ 32.4 + 20log10(f_GHz)
        """
        d = max(d_m, 1.0)
        shadow = float(self.rng.normal(0.0, self.cfg.shadowing_std_db))
        return self.pl0_db + 10.0 * self.cfg.pathloss_exp_n * math.log10(d) + shadow

    def fading_power_linear(self) -> float:
        if self.cfg.fading == "none":
            return 1.0
        if self.cfg.fading == "rayleigh":
            # |h|^2 ~ Exp(mean=1)
            return float(self.rng.exponential(1.0))
        raise ValueError("fading must be 'rayleigh' or 'none'")

    def inst_phy(self, ue: UE, bw_hz: float, p_rbg_w: float) -> Tuple[float, float, float]:
        """
        Returns: (rate_bps, snr_lin, noise_w)
        - Single cell => SINR == SNR (no inter-cell interference)
        - Uses calibrated pathloss in dB
        """
        # Noise
        n_w = noise_power_w(bw_hz, self.cfg.thermal_noise_dbm_hz, self.cfg.noise_figure_db)

        # Transmit power per RBG
        p_tx_dbm = w_to_dbm(p_rbg_w)

        # Pathloss
        pl_db = self.pathloss_db(ue.distance_m)

        # Small-scale fading (power)
        fad_lin = self.fading_power_linear()
        fad_db = linear_to_db(fad_lin)

        # Received power (dBm): Prx = Ptx - PL + fading_dB
        p_rx_dbm = p_tx_dbm - pl_db + fad_db

        # SNR(dB) = Prx(dBm) - N(dBm)
        n_dbm = noise_power_dbm(bw_hz, self.cfg.thermal_noise_dbm_hz, self.cfg.noise_figure_db)
        snr_db = p_rx_dbm - n_dbm
        snr_lin = db_to_linear(snr_db)

        # Shannon capacity
        rate_bps = bw_hz * math.log2(1.0 + snr_lin)
        return rate_bps, snr_lin, n_w

    # ---------- step ----------
    def step(self) -> None:
        dt = self.cfg.dt_s

        # Arrivals
        for ue in self.ues:
            bits_in = self.traffic[ue.slice_name].arrivals_bits(dt, self.rng)

            self.total_arrivals_bits[ue.slice_name] += bits_in
            self.ue_arrivals_bits[ue.ue_id] += bits_in

            room = self.cfg.buffer_max_bits - ue.buffer_bits
            accepted = max(0, min(bits_in, room))
            dropped = max(0, bits_in - accepted)

            ue.buffer_bits += accepted

            self.total_accepted_bits[ue.slice_name] += accepted
            self.total_dropped_bits[ue.slice_name] += dropped
            self.ue_accepted_bits[ue.ue_id] += accepted
            self.ue_dropped_bits[ue.ue_id] += dropped

        # Transmission
        if self.tx_on:
            slice_rbgs = self.rbg_allocation_by_slice()

            active_rbgs = sum(len(v) for v in slice_rbgs.values())
            if active_rbgs <= 0:
                active_rbgs = 1

            # Equal power per active RBG
            p_total_w = dbm_to_w(self.cfg.total_tx_power_dbm)
            p_per_rbg_w = p_total_w / active_rbgs

            ues_by_slice: Dict[str, List[UE]] = {s: [] for s in SLICES}
            for ue in self.ues:
                ues_by_slice[ue.slice_name].append(ue)

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

                    for ue in ues_s:
                        rate, snr_lin, n_w = self.inst_phy(ue, bw, p_per_rbg_w)
                        inst_rate[ue.ue_id] = rate
                        inst_snr[ue.ue_id] = snr_lin
                        inst_noise[ue.ue_id] = n_w

                    chosen = self.scheduler.pick(ues_s, inst_rate, self.rng)

                    # record PHY sample for chosen UE on this RBG
                    self.phy_window.add_sample(
                        slice_name=s,
                        ue_id=chosen.ue_id,
                        rate_bps=inst_rate[chosen.ue_id],
                        snr_lin=inst_snr[chosen.ue_id],
                        noise_w=inst_noise[chosen.ue_id],
                    )

                    cap_bits = int(math.floor(inst_rate[chosen.ue_id] * dt))
                    send_bits = min(chosen.buffer_bits, cap_bits)

                    chosen.buffer_bits -= send_bits
                    sent_bits_by_ue[chosen.ue_id] += send_bits
                    sent_bits_by_slice[s] += send_bits

            # totals
            for s in SLICES:
                self.total_sent_bits[s] += sent_bits_by_slice[s]
            for ue_id, b in sent_bits_by_ue.items():
                self.ue_sent_bits[ue_id] += b

            # PF update (uses actual per-step throughput)
            alpha = max(0.0, min(1.0, dt / max(self.cfg.pf_tc_s, 1e-9)))
            for ue in self.ues:
                inst_thr = sent_bits_by_ue[ue.ue_id] / dt
                ue.avg_thr_bps = (1 - alpha) * ue.avg_thr_bps + alpha * inst_thr

        self.t_s += dt

    # ---------- snapshots ----------
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


# -----------------------------
# Interactive control
# -----------------------------
class ControlThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.quit = False
        self.lines: List[str] = []
        self._lock = threading.Lock()

    def run(self):
        while not self.quit:
            line = sys.stdin.readline()
            if not line:
                self.quit = True
                return
            with self._lock:
                self.lines.append(line.rstrip("\n"))

    def pop_lines(self) -> List[str]:
        with self._lock:
            out = self.lines[:]
            self.lines.clear()
        return out


def mbps(bits: int, interval_s: float) -> float:
    return (bits / interval_s) / 1e6


def main():
    cfg = SimConfig()
    env = RANSandboxEnv(cfg)

    ctrl = ControlThread()
    ctrl.start()

    print("RAN sandbox started (REALISTIC pathloss @ 3.5GHz, 30 dBm total TX).")
    print("Controls:")
    print("  [Enter]                 toggle TX ON/OFF")
    print("  rr | pf | wf            set scheduler")
    print("  slice <e> <u>           set PRBs for eMBB and URLLC (mMTC gets remainder)")
    print("  detail on|off           per-UE PHY details each second")
    print("  status                  show current config")
    print("  q                       quit\n")

    report_period_s = 1.0
    next_report_t = report_period_s

    last_arr = {s: 0 for s in SLICES}
    last_acc = {s: 0 for s in SLICES}
    last_drp = {s: 0 for s in SLICES}
    last_snt = {s: 0 for s in SLICES}

    last_ue_arr = {ue.ue_id: 0 for ue in env.ues}
    last_ue_acc = {ue.ue_id: 0 for ue in env.ues}
    last_ue_drp = {ue.ue_id: 0 for ue in env.ues}
    last_ue_snt = {ue.ue_id: 0 for ue in env.ues}

    try:
        while not ctrl.quit:
            # commands
            for raw in ctrl.pop_lines():
                cmd = raw.strip()
                low = cmd.lower()

                if cmd == "":
                    env.tx_on = not env.tx_on
                    print(f"\n[MANUAL] TX toggled {'ON' if env.tx_on else 'OFF'}\n")
                    continue

                if low in ("q", "quit", "exit"):
                    ctrl.quit = True
                    break

                if low in ("rr", "pf", "wf"):
                    env.scheduler = make_scheduler(low)
                    print(f"\n[MANUAL] Scheduler set to {env.scheduler.name}\n")
                    continue

                if low.startswith("slice "):
                    parts = low.split()
                    if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                        e = int(parts[1]); u = int(parts[2])
                        env.set_slicing(e, u)
                        print(f"\n[MANUAL] Slicing set to {env.cfg.slice_prbs}\n")
                    else:
                        print("\nUsage: slice <eMBB_PRBs> <URLLC_PRBs>\n")
                    continue

                if low.startswith("detail "):
                    parts = low.split()
                    if len(parts) == 2 and parts[1] in ("on", "off"):
                        env.detail_on = (parts[1] == "on")
                        print(f"\n[MANUAL] detail {'ON' if env.detail_on else 'OFF'}\n")
                    else:
                        print("\nUsage: detail on|off\n")
                    continue

                if low == "status":
                    snap = env.snapshot()
                    print("\nSTATUS")
                    print("  TX:", "ON" if snap["tx_on"] else "OFF")
                    print("  Scheduler:", snap["scheduler"])
                    print("  Slice PRBs:", snap["slice_prbs"])
                    print("  Carrier:", f"{env.cfg.carrier_freq_ghz} GHz")
                    print("  Total TX:", f"{env.cfg.total_tx_power_dbm:.1f} dBm")
                    print("  UE distances (m):", {ue.ue_id: round(ue.distance_m, 2) for ue in env.ues})
                    print()
                    continue

                print("\nUnknown command.\n")

            env.step()

            if env.t_s + 1e-12 >= next_report_t:
                snap = env.snapshot()
                phy = env.pop_phy_window()
                ue_snap = env.ue_snapshot()

                # slice deltas
                arr = {s: snap["total_arrivals_bits"][s] - last_arr[s] for s in SLICES}
                acc = {s: snap["total_accepted_bits"][s] - last_acc[s] for s in SLICES}
                drp = {s: snap["total_dropped_bits"][s] - last_drp[s] for s in SLICES}
                snt = {s: snap["total_sent_bits"][s] - last_snt[s] for s in SLICES}

                last_arr = dict(snap["total_arrivals_bits"])
                last_acc = dict(snap["total_accepted_bits"])
                last_drp = dict(snap["total_dropped_bits"])
                last_snt = dict(snap["total_sent_bits"])

                print(f"t={snap['t_s']:.2f}s | TX={'ON' if snap['tx_on'] else 'OFF'} | sched={snap['scheduler']} | PRBs={snap['slice_prbs']}")
                print("  arrivals  (Mbps):", {s: round(mbps(arr[s], report_period_s), 3) for s in SLICES})
                print("  accepted  (Mbps):", {s: round(mbps(acc[s], report_period_s), 3) for s in SLICES})
                print("  dropped   (Mbps):", {s: round(mbps(drp[s], report_period_s), 3) for s in SLICES})
                print("  sent      (Mbps):", {s: round(mbps(snt[s], report_period_s), 3) for s in SLICES})
                print("  buffer   (Mbits):", {s: round(snap["slice_buffer_bits"][s] / 1e6, 3) for s in SLICES})

                # PHY: average per scheduled RBG rate in this 1s window
                slice_avg_rate_mbps = {}
                for s in SLICES:
                    if phy.slice_samples[s] > 0:
                        avg_rate = phy.slice_rate_sum_bps[s] / phy.slice_samples[s]
                        slice_avg_rate_mbps[s] = round(avg_rate / 1e6, 3)
                    else:
                        slice_avg_rate_mbps[s] = 0.0
                print("  PHY avg RBG rate (Mbps):", slice_avg_rate_mbps)

                if env.detail_on and snap["tx_on"]:
                    print("  Per-UE (last 1s window):")
                    for row in ue_snap:
                        uid = row["ue"]

                        ue_arr = row["arrivals_bits"] - last_ue_arr[uid]
                        ue_acc = row["accepted_bits"] - last_ue_acc[uid]
                        ue_drp = row["dropped_bits"] - last_ue_drp[uid]
                        ue_snt = row["sent_bits"] - last_ue_snt[uid]

                        last_ue_arr[uid] = row["arrivals_bits"]
                        last_ue_acc[uid] = row["accepted_bits"]
                        last_ue_drp[uid] = row["dropped_bits"]
                        last_ue_snt[uid] = row["sent_bits"]

                        samples = phy.ue_samples.get(uid, 0)
                        if samples > 0:
                            avg_rate = phy.ue_rate_sum_bps[uid] / samples
                            avg_snr_lin = phy.ue_snr_sum_lin[uid] / samples
                            avg_noise_w = phy.ue_noise_sum_w[uid] / samples
                            avg_snr_db = linear_to_db(avg_snr_lin)
                        else:
                            avg_rate = 0.0
                            avg_snr_lin = 0.0
                            avg_noise_w = 0.0
                            avg_snr_db = float("-inf")

                        print(
                            f"    UE{uid} ({row['slice']}, d={row['dist_m']}m) | "
                            f"arr={mbps(ue_arr,1):.3f}Mbps acc={mbps(ue_acc,1):.3f}Mbps drp={mbps(ue_drp,1):.3f}Mbps "
                            f"snt={mbps(ue_snt,1):.3f}Mbps | buf={row['buffer_bits']/1e6:.3f}Mbits | "
                            f"avgSNR={avg_snr_db:.2f}dB (lin={avg_snr_lin:.2e}) | "
                            f"avgNoise={avg_noise_w:.2e}W | avgRBGRate={avg_rate/1e6:.3f}Mbps"
                        )
                else:
                    # keep delta trackers consistent
                    for row in ue_snap:
                        uid = row["ue"]
                        last_ue_arr[uid] = row["arrivals_bits"]
                        last_ue_acc[uid] = row["accepted_bits"]
                        last_ue_drp[uid] = row["dropped_bits"]
                        last_ue_snt[uid] = row["sent_bits"]

                print()
                next_report_t += report_period_s

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting...")

    snap = env.snapshot()
    print("\nFinal summary:")
    print(f"Sim time: {snap['t_s']:.2f}s")
    print("Total arrivals  (Mbits):", {s: round(snap["total_arrivals_bits"][s] / 1e6, 3) for s in SLICES})
    print("Total accepted  (Mbits):", {s: round(snap["total_accepted_bits"][s] / 1e6, 3) for s in SLICES})
    print("Total dropped   (Mbits):", {s: round(snap["total_dropped_bits"][s] / 1e6, 3) for s in SLICES})
    print("Total sent      (Mbits):", {s: round(snap["total_sent_bits"][s] / 1e6, 3) for s in SLICES})
    print("Final buffer    (Mbits):", {s: round(snap["slice_buffer_bits"][s] / 1e6, 3) for s in SLICES})


if __name__ == "__main__":
    main()
