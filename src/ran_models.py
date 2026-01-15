import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Callable, Optional, List

import numpy as np


#seed setting function
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)

#base-2 log calculation.
def log2(x: float) -> float:
    return math.log(x, 2)


#Data Models

@dataclass
class User:
    id: int
    bs_id: int
    traffic_type: str
    arrival_rate: float          # bits/sec (Poisson mean rate)
    constant_rate: float         # bits/sec (CBR)
    position: Tuple[float, float]
    buffer: float = 0.0          # bits pending

    def generate_traffic_bits(self, slot_time: float) -> float:
        """
        Generates new demand arriving into the (downlink) buffer during this slot.
        Returns the number of bits that arrived this slot.
        """
        new_bits = 0.0

        # Constant bit rate
        if self.constant_rate > 0:
            new_bits += self.constant_rate * slot_time

        # Poisson arrivals (mean in bits)
        if self.arrival_rate > 0:
            mean_bits = self.arrival_rate * slot_time
            # Poisson expects "events"; we treat "bits" as the random variable unit.
            # This is okay for an abstract simulation.
            new_bits += float(np.random.poisson(mean_bits))

        self.buffer += new_bits
        return new_bits


@dataclass
class BaseStation:
    id: int
    position: Tuple[float, float]
    n_rbg: int
    total_power: float           # total BS power budget (arbitrary units)


# -------------------------
# Main Environment
# -------------------------

class RANEnvironment:
    """
    A simplified downlink RAN sandbox environment suitable for PACIFISTA-style profiling:
    - Users have downlink demand (buffer)
    - BS allocates RBGs (frequency chunks) and power to users each slot
    - Delivered bits computed using Shannon-like capacity: B * T * log2(1 + SINR)
    - Logs both inputs (arrivals, channels, allocations) and outputs (throughput, buffer, SINR)
    """
    #noise_power = 6.31e-12
    """
    # Thermal noise power calculation for 540 kHz RBG:
    # N0 = -174 dBm/Hz (Boltzmann constant at 290K, universal constant)
    # RBG bandwidth = 540 kHz = 540,000 Hz (3 resource blocks × 180 kHz)
    # Thermal noise over 540 kHz = -174 + 10*log10(540000) = -116.68 dBm
    # UE receiver noise figure (NF) = 5 dB (realistic 3GPP standard)
    # Total noise = -116.68 + 5 = -111.68 dBm ≈ -112 dBm
    # Linear form: 10^(-112/10) = 6.31e-12 mW
    # Used in SINR = signal / (interference + noise_power)
    # Realistic value; previous 2e-15 was 25 dB too high (unrealistic)
    """
    def __init__(
        self,
        num_bs: int = 1,
        num_users_per_bs: Optional[List[int]] = None,
        slot_time: float = 0.01,              # seconds
        n_rbg: int = 17,
        total_power: float = 1.0,
        noise_power: float = 6.31e-12,
        path_loss_exp: float = 3.5,
        cell_radius: float = 50.0,
        inter_bs_distance: float = 120.0,
        rbg_bandwidth_hz: float = 540e3,      # ~10 MHz / 17 RBG ≈ 540 kHz
        seed: int = 42,
        track_verbose_logs: bool = True,
    ):
        set_seed(seed)

        self.slot_time = float(slot_time)
        self.noise_power = float(noise_power)
        self.path_loss_exp = float(path_loss_exp)
        self.rbg_bandwidth_hz = float(rbg_bandwidth_hz)

        self.track_verbose_logs = track_verbose_logs

        # Base stations
        self.base_stations: List[BaseStation] = []
        for b in range(num_bs):
            pos = (b * inter_bs_distance, 0.0)
            self.base_stations.append(BaseStation(id=b, position=pos, n_rbg=n_rbg, total_power=total_power))

        # Users per BS
        if num_users_per_bs is None:
            num_users_per_bs = [3] * num_bs
        elif isinstance(num_users_per_bs, int):
            num_users_per_bs = [num_users_per_bs] * num_bs

        # Traffic settings (bits/sec). Matches typical PACIFISTA setup style.
        rate_settings = {
            "eMBB":  {"constant_rate": 4e6,      "arrival_rate": 0.0},
            "URLLC": {"constant_rate": 0.0,      "arrival_rate": 89.29e3},
            "mMTC":  {"constant_rate": 0.0,      "arrival_rate": 44.64e3},
        }
        default_cycle = ["eMBB", "URLLC", "mMTC"]

        # Build users
        self.users: List[User] = []
        self.users_by_bs: Dict[int, List[int]] = {bs.id: [] for bs in self.base_stations}

        uid = 0
        for bs in self.base_stations:
            b = bs.id
            for i in range(num_users_per_bs[b]):
                ttype = default_cycle[i % len(default_cycle)]
                rates = rate_settings[ttype]

                # Place UE uniformly in disk around BS
                angle = random.random() * 2 * math.pi
                r = cell_radius * math.sqrt(random.random())
                ux = bs.position[0] + r * math.cos(angle)
                uy = bs.position[1] + r * math.sin(angle)

                u = User(
                    id=uid,
                    bs_id=b,
                    traffic_type=ttype,
                    arrival_rate=float(rates["arrival_rate"]),
                    constant_rate=float(rates["constant_rate"]),
                    position=(ux, uy),
                    buffer=0.0
                )
                self.users.append(u)
                self.users_by_bs[b].append(uid)
                uid += 1

        # Channel gains: gain[bs_id][user_id]
        self.channel_gain: Dict[int, Dict[int, float]] = {bs.id: {} for bs in self.base_stations}
        for bs in self.base_stations:
            for u in self.users:
                dx = bs.position[0] - u.position[0]
                dy = bs.position[1] - u.position[1]
                d = math.sqrt(dx * dx + dy * dy)
                d = max(d, 1.0)  # min 1m
                self.channel_gain[bs.id][u.id] = 1.0 / (d ** self.path_loss_exp)

        # Policy (can be injected)
        self.policy: Optional[Callable[["RANEnvironment"], Dict[int, Dict[int, Tuple[int, float]]]]] = None

        # Master logs (per-slot)
        # These are the "PACIFISTA-style" raw time-series data you will use later for ECDF/K-S/INT.
        self.logs = {
            "arrivals_bits": {u.id: [] for u in self.users},      # input generated each slot
            "buffer_bits": {u.id: [] for u in self.users},        # buffer after TX each slot
            "delivered_bits": {u.id: [] for u in self.users},     # throughput per slot (bits)
            "capacity_bits": {u.id: [] for u in self.users},      # capacity per slot (bits)
            "sinr_linear": {u.id: [] for u in self.users},        # avg SINR per slot
            "rbg_count": {u.id: [] for u in self.users},          # how many RBG each UE got
            "total_delivered_bits": [],                           # system total per slot
        }

        # Verbose logs (optional): store allocations and powers per slot
        # This can be big, but it helps you "see everything".
        self.verbose = {
            "decisions": [],  # list of decisions dict per slot: {bs_id: {rbg_idx: (user_id, power)}}
        }

        self._slot_index = 0

    # -------------------------
    # Policies
    # -------------------------

    def set_policy(self, policy_func: Callable[["RANEnvironment"], Dict[int, Dict[int, Tuple[int, float]]]]) -> None:
        self.policy = policy_func

    def default_policy(self) -> Dict[int, Dict[int, Tuple[int, float]]]:
        """
        Simple round-robin per BS with equal power per RBG.
        Returns:
          decisions[bs_id][rbg_idx] = (user_id, power)
        """
        decisions: Dict[int, Dict[int, Tuple[int, float]]] = {}
        for bs in self.base_stations:
            decisions[bs.id] = {}
            uids = self.users_by_bs[bs.id]
            if not uids:
                continue

            power_per_rbg = bs.total_power / bs.n_rbg
            for rbg in range(bs.n_rbg):
                user_id = uids[rbg % len(uids)]
                decisions[bs.id][rbg] = (user_id, power_per_rbg)

        return decisions

    # -------------------------
    # Core Simulation
    # -------------------------

    def step(self, policy_func: Optional[Callable[["RANEnvironment"], Dict[int, Dict[int, Tuple[int, float]]]]] = None):
        """
        Run one slot:
          - Generate arrivals
          - Compute allocations (decisions)
          - Compute SINR + capacity + delivered bits
          - Update buffers
          - Log everything
        """
        slot = self._slot_index

        # (1) arrivals
        arrivals_this_slot = {}
        for u in self.users:
            a = u.generate_traffic_bits(self.slot_time)
            arrivals_this_slot[u.id] = a
            self.logs["arrivals_bits"][u.id].append(a)

        # (2) policy decisions
        if policy_func is not None:
            decisions = policy_func(self)
        elif self.policy is not None:
            decisions = self.policy(self)
        else:
            decisions = self.default_policy()

        # Ensure all BS keys exist
        for bs in self.base_stations:
            decisions.setdefault(bs.id, {})

        if self.track_verbose_logs:
            self.verbose["decisions"].append(decisions)

        # (3) simulate transmission
        delivered_bits = {u.id: 0.0 for u in self.users}
        capacity_bits = {u.id: 0.0 for u in self.users}
        sinr_samples = {u.id: [] for u in self.users}
        rbg_count = {u.id: 0 for u in self.users}

        # quick lookup
        user_map = {u.id: u for u in self.users}

        for bs in self.base_stations:
            alloc = decisions[bs.id]  # {rbg: (user_id, power)}
            for rbg_idx, (uid, pwr) in alloc.items():
                if rbg_idx < 0 or rbg_idx >= bs.n_rbg:
                    continue

                # Only allow BS to schedule its own attached UE (you can relax later)
                if user_map[uid].bs_id != bs.id:
                    continue

                rbg_count[uid] += 1

                # Signal at UE from serving BS
                signal = pwr * self.channel_gain[bs.id][uid]

                # Interference from other BS using same RBG index
                interf = 0.0
                for other_bs in self.base_stations:
                    if other_bs.id == bs.id:
                        continue
                    other_alloc = decisions[other_bs.id]
                    if rbg_idx in other_alloc:
                        _, other_pwr = other_alloc[rbg_idx]
                        interf += other_pwr * self.channel_gain[other_bs.id][uid]

                sinr = signal / (self.noise_power + interf)
                sinr_samples[uid].append(sinr)

                # Shannon capacity bits on this RBG in this slot
                cap = self.rbg_bandwidth_hz * self.slot_time * log2(1.0 + sinr)

                # Can’t send more than available buffer
                uobj = user_map[uid]
                send = min(uobj.buffer, cap)

                uobj.buffer -= send
                delivered_bits[uid] += send
                capacity_bits[uid] += cap

        # (4) per-slot logging
        total = 0.0
        for u in self.users:
            uid = u.id

            self.logs["delivered_bits"][uid].append(delivered_bits[uid])
            self.logs["capacity_bits"][uid].append(capacity_bits[uid])

            avg_sinr = float(np.mean(sinr_samples[uid])) if len(sinr_samples[uid]) > 0 else 0.0
            self.logs["sinr_linear"][uid].append(avg_sinr)

            self.logs["rbg_count"][uid].append(rbg_count[uid])
            self.logs["buffer_bits"][uid].append(u.buffer)

            total += delivered_bits[uid]

        self.logs["total_delivered_bits"].append(total)

        self._slot_index += 1
        return decisions

    def run(self, num_slots: int, policy_func: Optional[Callable[["RANEnvironment"], Dict[int, Dict[int, Tuple[int, float]]]]] = None):
        for _ in range(num_slots):
            self.step(policy_func)

    # -------------------------
    # Debug / Visibility
    # -------------------------

    def print_slot(self, slot_idx: int):
        """
        Print a full snapshot of what happened in one slot:
        - arrivals
        - allocations
        - SINR/capacity/delivered/buffer
        """
        if slot_idx < 0 or slot_idx >= len(self.logs["total_delivered_bits"]):
            raise ValueError("slot_idx out of range")

        print("=" * 90)
        print(f"SLOT {slot_idx}   (slot_time={self.slot_time}s)")
        print("-" * 90)

        # Arrivals
        print("ARRIVALS (generated input bits this slot):")
        for u in self.users:
            a = self.logs["arrivals_bits"][u.id][slot_idx]
            print(f"  UE{u.id:02d} ({u.traffic_type}) -> +{a:.0f} bits")

        # Decisions
        if self.track_verbose_logs and slot_idx < len(self.verbose["decisions"]):
            decisions = self.verbose["decisions"][slot_idx]
            print("\nALLOCATIONS (BS -> RBG -> UE, power):")
            for bs in self.base_stations:
                alloc = decisions.get(bs.id, {})
                print(f"  BS{bs.id}:")
                for rbg_idx in sorted(alloc.keys()):
                    uid, pwr = alloc[rbg_idx]
                    print(f"    RBG{rbg_idx:02d} -> UE{uid:02d}  P={pwr:.4g}")
        else:
            print("\nALLOCATIONS: (verbose logging disabled)")

        # Outcomes per user
        print("\nOUTCOMES (per UE):")
        for u in self.users:
            uid = u.id
            sinr = self.logs["sinr_linear"][uid][slot_idx]
            cap = self.logs["capacity_bits"][uid][slot_idx]
            tx = self.logs["delivered_bits"][uid][slot_idx]
            buf = self.logs["buffer_bits"][uid][slot_idx]
            rbgs = self.logs["rbg_count"][uid][slot_idx]
            print(
                f"  UE{uid:02d} rbgs={rbgs:2d}  SINR={sinr:.3g}  cap={cap:.0f} bits  "
                f"tx={tx:.0f} bits  buf={buf:.0f} bits"
            )

        print(f"\nTOTAL delivered bits this slot: {self.logs['total_delivered_bits'][slot_idx]:.0f}")
        print("=" * 90)

    def summary(self):
        """
        Prints a PACIFISTA-friendly summary:
        - avg throughput per UE (bits/s)
        - avg arrivals per UE (bits/s)
        - avg buffer
        """
        slots = len(self.logs["total_delivered_bits"])
        total_time = slots * self.slot_time
        print("\n--- SUMMARY ---")
        print(f"Slots: {slots}, slot_time={self.slot_time}s, simulated_time={total_time:.2f}s")

        for u in self.users:
            uid = u.id
            avg_arr = sum(self.logs["arrivals_bits"][uid]) / total_time
            avg_tx = sum(self.logs["delivered_bits"][uid]) / total_time
            avg_buf = float(np.mean(self.logs["buffer_bits"][uid]))
            print(
                f"UE{uid:02d} ({u.traffic_type})  "
                f"avg_arrivals={avg_arr:.2f} bits/s  avg_throughput={avg_tx:.2f} bits/s  avg_buffer={avg_buf:.2f} bits"
            )

        net_avg = sum(self.logs["total_delivered_bits"]) / total_time
        print(f"Network avg throughput: {net_avg:.2f} bits/s")

    def export_csv(self, prefix: str = "ran_logs"):
        """
        Export key logs to CSV for later ECDF/K-S/INT processing.
        Produces:
          - {prefix}_per_user.csv
          - {prefix}_total.csv
        """
        import csv

        slots = len(self.logs["total_delivered_bits"])
        per_user_path = f"{prefix}_per_user.csv"
        total_path = f"{prefix}_total.csv"

        # per-user
        with open(per_user_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "slot", "user_id", "traffic_type",
                "arrivals_bits", "delivered_bits", "capacity_bits", "buffer_bits",
                "sinr_linear", "rbg_count"
            ])
            for slot in range(slots):
                for u in self.users:
                    uid = u.id
                    writer.writerow([
                        slot, uid, u.traffic_type,
                        self.logs["arrivals_bits"][uid][slot],
                        self.logs["delivered_bits"][uid][slot],
                        self.logs["capacity_bits"][uid][slot],
                        self.logs["buffer_bits"][uid][slot],
                        self.logs["sinr_linear"][uid][slot],
                        self.logs["rbg_count"][uid][slot],
                    ])

        # total
        with open(total_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["slot", "total_delivered_bits"])
            for slot in range(slots):
                writer.writerow([slot, self.logs["total_delivered_bits"][slot]])

        print(f"Exported: {per_user_path}, {total_path}")


# -------------------------
# Example main
# -------------------------

if __name__ == "__main__":
    # Example: similar size to your run (2 BS, 3 users each = 6 UEs total)
    env = RANEnvironment(
        num_bs=1,
        num_users_per_bs=[6],
        slot_time=0.01,
        n_rbg=17,
        total_power=1.0,
        noise_power=2e-15,
        path_loss_exp=3.5,
        cell_radius=50.0,
        seed=42,
        track_verbose_logs=True,  # set False if memory gets large
    )

    num_slots = 200  # start small for visibility; increase to 1000+ later
    env.run(num_slots)

    env.summary()

    # Print a few example slots so you can "see all generated inputs"
    env.print_slot(0)
    env.print_slot(1)
    env.print_slot(2)

    # Export logs if you want to inspect in Excel / pandas
    env.export_csv(prefix="ran_logs")
