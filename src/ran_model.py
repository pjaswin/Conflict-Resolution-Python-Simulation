import math, random
import numpy as np

class User:
    def __init__(self, user_id, bs_id, traffic_type="eMBB", arrival_rate=0.0, constant_rate=0.0, position=(0,0)):
        """
        Represents a user equipment (UE) in the simulation.
        - user_id: unique identifier for the user.
        - bs_id: the id of the base station this user is attached to.
        - traffic_type: type of traffic ("eMBB", "URLLC", "mMTC", etc).
        - arrival_rate: average arrival rate in bits per second (for Poisson traffic).
        - constant_rate: constant bit rate traffic (bits per second).
        - position: (x,y) coordinates of the user (for path loss calculation).
        """
        self.id = user_id
        self.bs_id = bs_id
        self.traffic_type = traffic_type
        # Traffic generation parameters
        self.constant_rate = constant_rate
        self.arrival_rate = arrival_rate
        self.position = position
        self.buffer = 0  # bits currently in buffer waiting to be transmitted

    def generate_traffic(self, slot_time):
        """Generate new traffic for this user in this slot and add to buffer."""
        new_bits = 0
        if self.constant_rate > 0:
            # constant bit rate traffic
            new_bits += self.constant_rate * slot_time
        if self.arrival_rate > 0:
            # Poisson arrivals with mean arrival_rate * slot_time bits:contentReference[oaicite:14]{index=14}
            mean_bits = self.arrival_rate * slot_time
            if mean_bits > 0:
                # Use numpy for Poisson generation if available
                try:
                    new_bits += int(np.random.poisson(mean_bits))
                except Exception as e:
                    # Fallback: approximate Poisson by normal distribution for large mean
                    new_bits += max(0, int(random.gauss(mean_bits, math.sqrt(mean_bits))))
        # Update buffer with new arrivals
        self.buffer += new_bits
        return new_bits

class BaseStation:
    def __init__(self, bs_id, position=(0,0), n_rbg=17, total_power=1.0):
        """
        Represents a base station in the simulation.
        - bs_id: unique identifier for the base station.
        - position: (x,y) coordinates of the base station.
        - n_rbg: number of Resource Block Groups (RBGs) available at this BS.
        - total_power: total transmit power available per slot (W or relative units).
        """
        self.id = bs_id
        self.position = position
        self.n_rbg = n_rbg
        self.total_power = total_power

class RANEnvironment:
    def __init__(self, num_bs=1, num_users_per_bs=None, bs_positions=None,
                 cell_radius=50.0, inter_bs_distance=100.0,
                 traffic_types=None,
                 slot_time=0.01,
                 path_loss_exp=3.5,
                 n_rbg=17, total_power=1.0, noise_power=2e-15,
                 track_control_actions=True):
        """
        Initialize the RAN environment (sandbox) for simulation.

        - num_bs: number of base stations.
        - num_users_per_bs: list or int. If list, specifies number of UEs per BS; if int, uses same number for each BS.
        - bs_positions: list of (x,y) coordinates for each BS. If None, BS are placed along x-axis separated by inter_bs_distance.
        - cell_radius: radius within which UEs are placed around their serving BS (if positions are used).
        - inter_bs_distance: spacing between BS if bs_positions not provided.
        - traffic_types: structure defining each user's traffic type. 
                         If None, default cycle ["eMBB","URLLC","mMTC"] is used per BS.
                         Can be a list for each BS or a single combined list.
        - slot_time: duration of one time slot in seconds (e.g., 0.01 = 10ms).
        - path_loss_exp: path loss exponent for channel gain calculation.
        - n_rbg: number of RBGs (resource block groups) per base station.
        - total_power: total transmit power per base station (to be divided among RBGs).
        - noise_power: noise power per RBG over one time slot (W).
        - track_control_actions: whether to log control actions (resource allocations) for conflict analysis.
        """
        self.num_bs = num_bs
        self.slot_time = slot_time
        self.path_loss_exp = path_loss_exp
        self.noise_power = noise_power
        # Initialize base stations with given or default positions
        self.base_stations = []
        if bs_positions is None:
            bs_positions = [(i * inter_bs_distance, 0) for i in range(num_bs)]
        for i in range(num_bs):
            pos = bs_positions[i] if i < len(bs_positions) else (i * inter_bs_distance, 0)
            bs = BaseStation(i, position=pos, n_rbg=n_rbg, total_power=total_power)
            self.base_stations.append(bs)
        # Initialize users and assign to base stations
        self.users = []
        self.users_by_bs = {bs.id: [] for bs in self.base_stations}
        user_id_counter = 0
        # Determine number of users per BS
        if num_users_per_bs is None:
            num_users_list = [2] * num_bs
        elif isinstance(num_users_per_bs, int):
            num_users_list = [num_users_per_bs] * num_bs
        else:
            num_users_list = num_users_per_bs
        # Determine traffic types for each user
        default_cycle = ["eMBB", "URLLC", "mMTC"]
        if traffic_types is None:
            # Use default cycle of types for each BS
            traffic_types = []
            for b in range(num_bs):
                types_for_bs = []
                for u in range(num_users_list[b]):
                    types_for_bs.append(default_cycle[u % len(default_cycle)])
                traffic_types.append(types_for_bs)
        else:
            # Normalize traffic_types to list of lists per BS
            if isinstance(traffic_types[0], str):
                # Single list for all users; partition it for each BS
                flat_types = traffic_types
                traffic_types = []
                idx = 0
                for b in range(num_bs):
                    types_for_bs = []
                    for u in range(num_users_list[b]):
                        if idx < len(flat_types):
                            types_for_bs.append(flat_types[idx])
                        else:
                            types_for_bs.append(default_cycle[u % len(default_cycle)])
                        idx += 1
                    traffic_types.append(types_for_bs)
        # Traffic rate settings (bits/sec) for each traffic type, based on PACIFISTA experiment:contentReference[oaicite:15]{index=15}
        rate_settings = {
            "eMBB": {"constant_rate": 4e6, "arrival_rate": 0},      # 4 Mbps constant traffic:contentReference[oaicite:16]{index=16}
            "URLLC": {"constant_rate": 0, "arrival_rate": 89.29e3}, # ~89.29 kbps Poisson traffic:contentReference[oaicite:17]{index=17}
            "mMTC": {"constant_rate": 0, "arrival_rate": 44.64e3}   # ~44.64 kbps Poisson traffic:contentReference[oaicite:18]{index=18}
        }
        # Create users for each BS
        for bs in self.base_stations:
            b = bs.id
            num_u = num_users_list[b] if b < len(num_users_list) else 0
            for u in range(num_u):
                ttype = traffic_types[b][u] if b < len(traffic_types) and u < len(traffic_types[b]) else default_cycle[u % len(default_cycle)]
                # Place user randomly within cell_radius of BS
                angle = random.random() * 2 * math.pi
                r = cell_radius * math.sqrt(random.random())
                ux = bs.position[0] + r * math.cos(angle)
                uy = bs.position[1] + r * math.sin(angle)
                rates = rate_settings.get(ttype, {"constant_rate": 0, "arrival_rate": 0})
                user = User(user_id_counter, bs_id=b, traffic_type=ttype,
                            arrival_rate=rates["arrival_rate"], constant_rate=rates["constant_rate"],
                            position=(ux, uy))
                self.users.append(user)
                self.users_by_bs[b].append(user)
                user_id_counter += 1
        # Pre-compute channel gains from each BS to each user for SINR calculation
        self.channel_gain = {bs.id: {} for bs in self.base_stations}
        for bs in self.base_stations:
            for user in self.users:
                dx = bs.position[0] - user.position[0]
                dy = bs.position[1] - user.position[1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < 1.0:
                    d = 1.0  # min distance 1m
                gain = 1.0 / (d ** self.path_loss_exp)
                self.channel_gain[bs.id][user.id] = gain
        # Initialize data structures for logging Key Performance Metrics (KPMs)
        self.track_control = track_control_actions
        # Key metrics (KPMs) include throughput, buffer occupancy, SINR, etc:contentReference[oaicite:19]{index=19}
        # KPMs per iteration: throughput per user, buffer per user, SINR per user, and total throughput
        self.kpm_history = {
            "throughput_per_user": {user.id: [] for user in self.users},
            "buffer_per_user": {user.id: [] for user in self.users},
            "sinr_per_user": {user.id: [] for user in self.users},
            "total_throughput": []
        }
        if self.track_control:
            # Also log control actions: number of RBGs allocated to each user per slot
            self.kpm_history["rbg_allocations_per_user"] = {user.id: [] for user in self.users}
        # Optional policy callback (if set via set_policy)
        self.policy = None

    def set_policy(self, policy_func):
        """Set a default policy function for resource allocation (xApp logic) to be used each slot."""
        self.policy = policy_func

    def default_policy(self):
        """
        Default scheduling policy (used if no external policy is provided).
        This simple policy allocates each RBG to users in a round-robin fashion per BS, with equal power split.
        """
        decisions = {}
        for bs in env.base_stations:
            decisions[bs.id] = {}
            if len(env.users_by_bs[bs.id]) == 0:
                continue
            # Round-robin through users for each RBG
            for rbg_idx in range(bs.n_rbg):
                user_list = env.users_by_bs[bs.id]
                user = user_list[rbg_idx % len(user_list)]
                # Allocate equal power share if multiple RBGs; here we set power per RBG = total_power / (number of RBGs allocated by this BS)
                # We don't know total count a priori in this simple scheme (could be all RBGs), but we can just assume full reuse (all RBGs used).
                p = bs.total_power / bs.n_rbg
                decisions[bs.id][rbg_idx] = (user.id, p)
        return decisions

    def step(self, policy_func=None):
        """
        Simulate one slot:
         - Generates new traffic arrivals (Poisson/constant) for each user.
         - Applies the control policy to decide resource (RBG) allocations and power per BS.
         - Simulates data transmission for this slot and updates user buffers.
         - Logs KPMs (throughput, buffer, SINR, etc.) for this iteration.

        policy_func: optional function env -> decisions dict. If None, uses self.policy or default_policy.
        Returns: decisions used (for reference).
        """
        # 1. Traffic arrivals
        for user in self.users:
            user.generate_traffic(self.slot_time)
        # 2. Determine resource allocation decisions via policy
        if policy_func is None:
            if self.policy is not None:
                policy_func = self.policy
            else:
                policy_func = RANEnvironment.default_policy  # use class default
        decisions = policy_func(self)
        # Ensure every BS key is present in decisions
        for bs in self.base_stations:
            if bs.id not in decisions:
                decisions[bs.id] = {}
        # 3. Simulate transmission based on decisions
        # Prepare variables to accumulate metrics
        user_throughputs = {user.id: 0.0 for user in self.users}
        user_sinrs = {user.id: [] for user in self.users}
        user_rbg_count = {user.id: 0 for user in self.users} if self.track_control else None
        # Map user id to user object for quick access
        user_map = {user.id: user for user in self.users}
        # Calculate delivered bits for each user
        for bs in self.base_stations:
            alloc = decisions.get(bs.id, {})
            # If BS allocated any RBGs this slot
            for rbg_idx, (user_id, p) in alloc.items():
                # Validate RBG index
                if rbg_idx < 0 or rbg_idx >= bs.n_rbg:
                    continue  # skip invalid
                # Ensure user is served by this BS (avoid misallocations)
                if user_map[user_id].bs_id != bs.id:
                    continue
                # If power not specified or invalid, use equal share of total_power
                power = p if p is not None else (bs.total_power / len(alloc) if len(alloc)>0 else 0)
                # Count allocation for control logging
                if self.track_control:
                    user_rbg_count[user_id] += 1
                # Compute SINR for this user on this RBG
                signal_power = power * self.channel_gain[bs.id][user_id]
                interference_power = 0.0
                # Interference from other BS using the same RBG
                for other_bs in self.base_stations:
                    if other_bs.id == bs.id:
                        continue
                    other_alloc = decisions.get(other_bs.id, {})
                    if rbg_idx in other_alloc:
                        _, other_p = other_alloc[rbg_idx]
                        if other_p is None:
                            other_p = other_bs.total_power / (len(other_alloc) if len(other_alloc)>0 else 1)
                        interference_power += other_p * self.channel_gain[other_bs.id].get(user_id, 0)
                sinr = signal_power / (self.noise_power + interference_power)
                # Record SINR (store linear or dB? - we'll store linear value here)
                user_sinrs[user_id].append(sinr)
                # Calculate achievable bits on this RBG during the slot (Shannon capacity)
                # bits = bandwidth(Hz) * slot_time(s) * log2(1+SINR). If using RBG of 540 kHz as in 10MHz/17 RBG.
                # We assume each RBG spans approx 540 kHz as in 10 MHz channel:contentReference[oaicite:20]{index=20}.
                bandwidth = 540e3  # 540 kHz per RBG
                bits_capacity = bandwidth * self.slot_time * math.log2(1 + sinr)
                # Determine bits transmitted (cannot exceed what is in buffer)
                user_obj = user_map[user_id]
                bits_to_send = min(user_obj.buffer, bits_capacity)
                user_obj.buffer -= bits_to_send  # remove sent bits from buffer
                user_throughputs[user_id] += bits_to_send
            # (end loop over RBGs for this BS)
        # end loop over BS
        # 4. Log the metrics for this slot
        total_bits = 0.0
        for user in self.users:
            uid = user.id
            # Throughput: bits successfully delivered this slot
            thr = user_throughputs[uid]
            total_bits += thr
            self.kpm_history["throughput_per_user"][uid].append(thr)
            # Buffer status: bits remaining in buffer after transmission
            self.kpm_history["buffer_per_user"][uid].append(user.buffer)
            # Average SINR experienced by this user (if multiple RBGs, take average of linear values)
            if user_sinrs[uid]:
                avg_sinr = sum(user_sinrs[uid]) / len(user_sinrs[uid])
            else:
                avg_sinr = 0.0
            self.kpm_history["sinr_per_user"][uid].append(avg_sinr)
            if self.track_control:
                self.kpm_history["rbg_allocations_per_user"][uid].append(user_rbg_count[uid])
        # total network throughput this slot
        self.kpm_history["total_throughput"].append(total_bits)
        return decisions

    def run(self, num_slots, policy_func=None):
        """
        Run the simulation for the given number of slots.
        Optionally specify a policy function; if None, uses self.policy or default.
        """
        for t in range(num_slots):
            self.step(policy_func)

# Example usage
if __name__ == "__main__":
    # Setup an environment with 2 base stations and 3 users each (mix of eMBB, URLLC, mMTC)
    env = RANEnvironment(num_bs=2, num_users_per_bs=[3,3])
    # Optionally, one can define a custom policy; here we use default for demonstration.
    # Run the simulation for a number of slots (e.g., 1000 slots which is 1000 * slot_time seconds)
    num_slots = 1000
    env.run(num_slots)
    # After simulation, compute and print some summary statistics.
    print("Simulation finished for", num_slots, "slots (slot_time =", env.slot_time, "s).")
    # Calculate average throughput per user (in bits per second)
    user_avg_throughput = {}
    for user in env.users:
        uid = user.id
        total_bits = sum(env.kpm_history["throughput_per_user"][uid])
        avg_rate = total_bits / (num_slots * env.slot_time)
        user_avg_throughput[uid] = avg_rate
        print(f"User {uid} ({user.traffic_type}) average throughput: {avg_rate:.2f} bits/s")
    # Total network throughput (bits/s)
    total_network_bits = sum(env.kpm_history["total_throughput"])
    total_network_rate = total_network_bits / (num_slots * env.slot_time)
    print(f"Overall network average throughput: {total_network_rate:.2f} bits/s")
