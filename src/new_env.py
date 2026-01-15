"""
This code is based on https://chatgpt.com/c/696419ed-8b5c-832d-b875-3b5a87ebf49a. New structure from ran_models.py. Felt better
"""

from __future__ import annotations

import math
import sys
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

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
# UE / config
# -----------------------------
@dataclass
class UE:
    ue_id: int
    slice_name: str
    distance_m: float
    buffer_bits: int = 0
    avg_thr_bps: float = 1.0  # for PF


@dataclass
class SimConfig:
    # Topology
    n_ues: int = 6
    cell_radius_m: float = 50.0

    # Radio resources
    n_prbs_total: int = 50
    n_rbgs: int = 17
    prb_bw_hz: float = 180e3  # simplified

    # Time
    dt_s: float = 0.01  # 10 ms

    # Buffers
    buffer_max_bits: int = 5_000_000  # per UE

    # Initial slicing PRBs
    slice_prbs: Dict[str, int] = field(default_factory=lambda: {"eMBB": 23, "URLLC": 16, "mMTC": 11})

    # Scheduler RR / PF / WF
    scheduler: str = "PF"

    # Power + noise
    total_tx_power_w: float = 1.0
    noise_figure_db: float = 7.0
    thermal_noise_dbm_hz: float = -174.0

    # Channel model
    pathloss_exp: float = 3.3
    shadowing_std_db: float = 6.0
    fading: str = "rayleigh"  # "rayleigh" or "none"

    # PF averaging time constant
    pf_tc_s: float = 1.0

    seed: int = 7


# -----------------------------
# RBG partition: 50 PRBs -> 17 RBGs
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
    """
    Approx "WF" as opportunistic scheduling: pick UE with best instantaneous rate.
    """
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
# Radio helpers
# -----------------------------
def db_to_linear(db: float) -> float:
    return 10 ** (db / 10.0)

def linear_to_db(x: float) -> float:
    return 10.0 * math.log10(max(x, 1e-30))

def noise_power_w(bw_hz: float, thermal_noise_dbm_hz: float, noise_figure_db: float) -> float:
    noise_dbm = thermal_noise_dbm_hz + 10 * math.log10(bw_hz) + noise_figure_db
    return 10 ** ((noise_dbm - 30.0) / 10.0)


# -----------------------------
# Per-step PHY stats container (aggregated)
# -----------------------------
@dataclass
class PhyAgg:
    # Aggregate per UE over the step (sum over its scheduled RBGs)
    ue_rate_sum_bps: Dict[int, float] = field(default_factory=dict)
    ue_snr_sum_lin: Dict[int, float] = field(default_factory=dict)
    ue_noise_sum_w: Dict[int, float] = field(default_factory=dict)
    ue_samples: Dict[int, int] = field(default_factory=dict)

    # Aggregate per slice over the step (sum of RBG rates)
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

    # control
    tx_on: bool = False
    scheduler: SchedulerBase = field(init=False)
    detail_on: bool = True  # print per-UE PHY details in report

    # time
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

    # last-interval PHY aggregation (we accumulate per step, then report per second)
    phy_window: PhyAgg = field(default_factory=PhyAgg)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.cfg.seed)

        # Assign 2 UEs per slice (default)
        slice_assign = ["eMBB", "eMBB", "URLLC", "URLLC", "mMTC", "mMTC"]
        if self.cfg.n_ues != 6:
            slice_assign = [SLICES[i % len(SLICES)] for i in range(self.cfg.n_ues)]

        self.ues = []
        for i in range(self.cfg.n_ues):
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

        # normalize slicing
        self.set_slicing(self.cfg.slice_prbs.get("eMBB", 0), self.cfg.slice_prbs.get("URLLC", 0))

        # init phy window
        self.phy_window.init_ues([ue.ue_id for ue in self.ues])

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

    # ---------- channel ----------
    def channel_gain_linear(self, ue: UE) -> float:
        # pathloss gain ~ d^{-alpha} * shadowing * fading
        shadow_db = float(self.rng.normal(0.0, self.cfg.shadowing_std_db))
        shadow_lin = db_to_linear(shadow_db)
        pl = shadow_lin * (ue.distance_m ** (-self.cfg.pathloss_exp))

        if self.cfg.fading == "none":
            fad = 1.0
        elif self.cfg.fading == "rayleigh":
            fad = float(self.rng.exponential(1.0))  # |h|^2
        else:
            raise ValueError("fading must be 'rayleigh' or 'none'")

        return pl * fad

    def inst_phy(self, ue: UE, bw_hz: float, p_w: float) -> Tuple[float, float, float]:
        """
        Returns: (rate_bps, snr_lin, noise_w)
        Single cell => SINR == SNR (no inter-cell interference).
        """
        n_w = noise_power_w(bw_hz, self.cfg.thermal_noise_dbm_hz, self.cfg.noise_figure_db)
        g = self.channel_gain_linear(ue)
        snr = (p_w * g) / max(n_w, 1e-30)
        rate = bw_hz * math.log2(1.0 + snr)
        return rate, snr, n_w

    # ---------- step ----------
    def step(self) -> None:
        dt = self.cfg.dt_s

        # 1) arrivals + accepted/dropped (per UE and per slice)
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

        # 2) transmission via PRB/RBG + SNR
        if self.tx_on:
            slice_rbgs = self.rbg_allocation_by_slice()

            # equal power per active RBG
            active_rbgs = sum(len(v) for v in slice_rbgs.values())
            if active_rbgs <= 0:
                active_rbgs = 1
            p_per_rbg = self.cfg.total_tx_power_w / active_rbgs

            ues_by_slice: Dict[str, List[UE]] = {s: [] for s in SLICES}
            for ue in self.ues:
                ues_by_slice[ue.slice_name].append(ue)

            sent_bits_by_ue: Dict[int, int] = {ue.ue_id: 0 for ue in self.ues}
            sent_bits_by_slice: Dict[str, int] = {s: 0 for s in SLICES}

            # ensure phy window init
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
                        rate, snr_lin, n_w = self.inst_phy(ue, bw, p_per_rbg)
                        inst_rate[ue.ue_id] = rate
                        inst_snr[ue.ue_id] = snr_lin
                        inst_noise[ue.ue_id] = n_w

                    chosen = self.scheduler.pick(ues_s, inst_rate, self.rng)

                    # PHY stats for chosen UE on this RBG
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

            # update totals
            for s in SLICES:
                self.total_sent_bits[s] += sent_bits_by_slice[s]
            for ue_id, b in sent_bits_by_ue.items():
                self.ue_sent_bits[ue_id] += b

            # PF averaging update using actual sent bits
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
        # return and reset aggregation window (per-second report window)
        out = self.phy_window
        self.phy_window = PhyAgg()
        self.phy_window.init_ues([ue.ue_id for ue in self.ues])
        return out


# -----------------------------
# Interactive control thread
# -----------------------------
class ControlThread(threading.Thread):
    """
    Commands:
      - Enter: toggle TX
      - rr|pf|wf: scheduler
      - slice <e> <u>: PRBs for eMBB and URLLC
      - detail on/off
      - status
      - q
    """
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


# -----------------------------
# Main runner
# -----------------------------
def main():
    cfg = SimConfig()
    env = RANSandboxEnv(cfg)

    ctrl = ControlThread()
    ctrl.start()

    print("RAN sandbox started (PRB/RBG + SNR + noise + capacity + per-UE stats).")
    print("Controls:")
    print("  [Enter]                 toggle TX ON/OFF")
    print("  rr | pf | wf            set scheduler")
    print("  slice <e> <u>           set PRBs for eMBB and URLLC (mMTC gets remainder)")
    print("  detail on|off           per-UE PHY details each second")
    print("  status                  print current config")
    print("  q                       quit\n")

    report_period_s = 1.0
    next_report_t = report_period_s

    # last totals for per-second deltas
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
            # Commands
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
                    print("  UE distances (m):", {ue.ue_id: round(ue.distance_m, 2) for ue in env.ues})
                    print()
                    continue

                print("\nUnknown command.\n")

            # step simulation
            env.step()

            # report each 1 second of simulated time
            if env.t_s + 1e-12 >= next_report_t:
                snap = env.snapshot()
                phy = env.pop_phy_window()
                ue_snap = env.ue_snapshot()

                # per-second slice deltas
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

                # slice PHY (capacity-ish): average scheduled RBG rate this window
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
                        # per-UE deltas in the window
                        ue_arr = row["arrivals_bits"] - last_ue_arr[uid]
                        ue_acc = row["accepted_bits"] - last_ue_acc[uid]
                        ue_drp = row["dropped_bits"] - last_ue_drp[uid]
                        ue_snt = row["sent_bits"] - last_ue_snt[uid]

                        last_ue_arr[uid] = row["arrivals_bits"]
                        last_ue_acc[uid] = row["accepted_bits"]
                        last_ue_drp[uid] = row["dropped_bits"]
                        last_ue_snt[uid] = row["sent_bits"]

                        # PHY averages for UE if it got scheduled
                        samples = phy.ue_samples.get(uid, 0)
                        if samples > 0:
                            avg_rate = phy.ue_rate_sum_bps[uid] / samples
                            avg_snr_lin = phy.ue_snr_sum_lin[uid] / samples
                            avg_noise_w = phy.ue_noise_sum_w[uid] / samples
                            avg_snr_db = linear_to_db(avg_snr_lin)
                        else:
                            avg_rate = 0.0
                            avg_snr_lin = 0.0
                            avg_snr_db = float("-inf")
                            avg_noise_w = 0.0

                        print(
                            f"    UE{uid} ({row['slice']}, d={row['dist_m']}m) | "
                            f"arr={mbps(ue_arr,1):.3f}Mbps acc={mbps(ue_acc,1):.3f}Mbps drp={mbps(ue_drp,1):.3f}Mbps "
                            f"snt={mbps(ue_snt,1):.3f}Mbps | buf={row['buffer_bits']/1e6:.3f}Mbits | "
                            f"avgSNR={avg_snr_db:.2f}dB (lin={avg_snr_lin:.2e}) | "
                            f"avgNoise={avg_noise_w:.2e}W | avgRBGRate={avg_rate/1e6:.3f}Mbps"
                        )
                else:
                    # Still keep per-UE delta trackers consistent when detail is off or TX off
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
