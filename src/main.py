"""
Main runner for the RAN sandbox (interactive + CSV logging).

Run:
  python main.py

CSV output:
  output/ran_timeseries.csv   (one row per 1-second report)
"""

from __future__ import annotations

import csv
import os
import sys
import time
import threading
from typing import Dict, List

from ran_environment import (
    SLICES,
    SimConfig,
    RANSandboxEnv,
    make_scheduler,
    linear_to_db,
)

# -----------------------------
# Helpers used only by main/logging
# -----------------------------
def mbps(bits: int, interval_s: float) -> float:
    return (bits / interval_s) / 1e6

def most_common_cqi(hist: Dict[int, int]):
    if not hist:
        return (-1, 0)
    cqi = max(hist.items(), key=lambda kv: kv[1])[0]
    return (cqi, hist[cqi])

def flatten_dict(prefix: str, d: Dict) -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        out[f"{prefix}{k}"] = v
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


def main():
    cfg = SimConfig()
    env = RANSandboxEnv(cfg)

    ctrl = ControlThread()
    ctrl.start()

    print("RAN sandbox started (REALISTIC pathloss @ 3.5GHz + SIMPLE MCS TABLE).")
    print("Controls:")
    print("  [Enter]                 toggle TX ON/OFF")
    print("  rr | pf | wf            set scheduler")
    print("  slice <e> <u>           set PRBs for eMBB and URLLC (mMTC gets remainder)")
    print("  detail on|off           per-UE PHY details each second")
    print("  status                  show current config")
    print("  q                       quit\n")

    # -------- CSV setup (same path as your original code) --------
    csv_path = "output/ran_timeseries.csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)  # avoid "No such file" error
    f_csv = open(csv_path, "w", newline="")
    writer: csv.DictWriter | None = None

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

                arr_bits = {s: snap["total_arrivals_bits"][s] - last_arr[s] for s in SLICES}
                acc_bits = {s: snap["total_accepted_bits"][s] - last_acc[s] for s in SLICES}
                drp_bits = {s: snap["total_dropped_bits"][s] - last_drp[s] for s in SLICES}
                snt_bits = {s: snap["total_sent_bits"][s] - last_snt[s] for s in SLICES}

                last_arr = dict(snap["total_arrivals_bits"])
                last_acc = dict(snap["total_accepted_bits"])
                last_drp = dict(snap["total_dropped_bits"])
                last_snt = dict(snap["total_sent_bits"])

                print(f"t={snap['t_s']:.2f}s | TX={'ON' if snap['tx_on'] else 'OFF'} | sched={snap['scheduler']} | PRBs={snap['slice_prbs']}")
                print("  arrivals  (Mbps):", {s: round(mbps(arr_bits[s], report_period_s), 3) for s in SLICES})
                print("  accepted  (Mbps):", {s: round(mbps(acc_bits[s], report_period_s), 3) for s in SLICES})
                print("  dropped   (Mbps):", {s: round(mbps(drp_bits[s], report_period_s), 3) for s in SLICES})
                print("  sent      (Mbps):", {s: round(mbps(snt_bits[s], report_period_s), 3) for s in SLICES})
                print("  buffer   (Mbits):", {s: round(snap["slice_buffer_bits"][s] / 1e6, 3) for s in SLICES})

                slice_avg_rate_mbps = {}
                for s in SLICES:
                    if phy.slice_samples[s] > 0:
                        avg_rate = phy.slice_rate_sum_bps[s] / phy.slice_samples[s]
                        slice_avg_rate_mbps[s] = round(avg_rate / 1e6, 3)
                    else:
                        slice_avg_rate_mbps[s] = 0.0
                print("  PHY avg RBG rate (Mbps):", slice_avg_rate_mbps)

                # ---------- CSV row ----------
                row: Dict[str, object] = {}
                row["t_s"] = float(snap["t_s"])
                row["tx_on"] = 1 if snap["tx_on"] else 0
                row["sched"] = snap["scheduler"]
                row.update(flatten_dict("prbs_", snap["slice_prbs"]))

                row.update({f"arr_mbps_{s}": mbps(arr_bits[s], report_period_s) for s in SLICES})
                row.update({f"acc_mbps_{s}": mbps(acc_bits[s], report_period_s) for s in SLICES})
                row.update({f"drp_mbps_{s}": mbps(drp_bits[s], report_period_s) for s in SLICES})
                row.update({f"snt_mbps_{s}": mbps(snt_bits[s], report_period_s) for s in SLICES})
                row.update({f"buf_mbits_{s}": snap["slice_buffer_bits"][s] / 1e6 for s in SLICES})
                row.update({f"phy_rbg_mbps_{s}": float(slice_avg_rate_mbps[s]) for s in SLICES})

                for ueinfo in ue_snap:
                    uid = ueinfo["ue"]

                    ue_arr = ueinfo["arrivals_bits"] - last_ue_arr[uid]
                    ue_acc = ueinfo["accepted_bits"] - last_ue_acc[uid]
                    ue_drp = ueinfo["dropped_bits"] - last_ue_drp[uid]
                    ue_snt = ueinfo["sent_bits"] - last_ue_snt[uid]

                    last_ue_arr[uid] = ueinfo["arrivals_bits"]
                    last_ue_acc[uid] = ueinfo["accepted_bits"]
                    last_ue_drp[uid] = ueinfo["dropped_bits"]
                    last_ue_snt[uid] = ueinfo["sent_bits"]

                    row[f"ue{uid}_slice"] = ueinfo["slice"]
                    row[f"ue{uid}_dist_m"] = float(ueinfo["dist_m"])
                    row[f"ue{uid}_buf_mbits"] = ueinfo["buffer_bits"] / 1e6

                    row[f"ue{uid}_arr_mbps"] = mbps(ue_arr, 1.0)
                    row[f"ue{uid}_acc_mbps"] = mbps(ue_acc, 1.0)
                    row[f"ue{uid}_drp_mbps"] = mbps(ue_drp, 1.0)
                    row[f"ue{uid}_snt_mbps"] = mbps(ue_snt, 1.0)

                    samples = phy.ue_samples.get(uid, 0)
                    if samples > 0:
                        avg_rate = phy.ue_rate_sum_bps[uid] / samples
                        avg_snr_lin = phy.ue_snr_sum_lin[uid] / samples
                        avg_noise_w = phy.ue_noise_sum_w[uid] / samples
                        avg_snr_db = linear_to_db(avg_snr_lin)
                        cqi, _cnt = most_common_cqi(phy.ue_cqi_hist.get(uid, {}))
                    else:
                        avg_rate = 0.0
                        avg_snr_db = float("nan")
                        avg_noise_w = float("nan")
                        cqi = -1

                    row[f"ue{uid}_avg_snr_db"] = avg_snr_db
                    row[f"ue{uid}_avg_noise_w"] = avg_noise_w
                    row[f"ue{uid}_avg_rbg_rate_mbps"] = avg_rate / 1e6
                    row[f"ue{uid}_cqi_mode"] = cqi

                if writer is None:
                    writer = csv.DictWriter(f_csv, fieldnames=list(row.keys()))
                    writer.writeheader()

                writer.writerow(row)
                f_csv.flush()
                os.fsync(f_csv.fileno())

                # Console per-UE details
                if env.detail_on and snap["tx_on"]:
                    print("  Per-UE (last 1s window):")
                    for u in ue_snap:
                        uid = u["ue"]
                        samples = phy.ue_samples.get(uid, 0)
                        if samples > 0:
                            avg_rate = phy.ue_rate_sum_bps[uid] / samples
                            avg_snr_lin = phy.ue_snr_sum_lin[uid] / samples
                            avg_noise_w = phy.ue_noise_sum_w[uid] / samples
                            avg_snr_db = linear_to_db(avg_snr_lin)
                            cqi, _cnt = most_common_cqi(phy.ue_cqi_hist.get(uid, {}))
                            note = ""
                        else:
                            avg_rate = 0.0
                            avg_snr_db = float("-inf")
                            avg_noise_w = 0.0
                            cqi = -1
                            note = " (not scheduled)"

                        print(
                            f"    UE{uid} ({u['slice']}, d={u['dist_m']}m) | "
                            f"buf={u['buffer_bits']/1e6:.3f}Mbits | "
                            f"avgSNR={avg_snr_db:.2f}dB | avgNoise={avg_noise_w:.2e}W | "
                            f"avgRBGRate={avg_rate/1e6:.3f}Mbps | CQI~{cqi}{note}"
                        )

                print()
                next_report_t += report_period_s

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received. Exiting...")

    finally:
        f_csv.close()
        print(f"\nSaved CSV time series to: {csv_path}")

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
