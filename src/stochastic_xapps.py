# stochastic_xapps.py
"""
Stochastic xApps (x1-x5) for PACIFISTA conflict evaluation.

Each xApp generates slicing policies based on Gaussian distributions.
Every 250ms, each xApp outputs a new PRB allocation policy.

This module is integrated with the RAN sandbox to test conflicts
in a realistic environment with traffic, PHY, and scheduling.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import defaultdict


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SlicePolicy:
    """Represents PRB allocation policy for each slice"""
    eMBB: int      # Enhanced Mobile Broadband
    URLLC: int     # Ultra-Reliable Low-Latency Communication
    mMTC: int      # Massive Machine-Type Communication
    timestamp: float


@dataclass
class xAppConfig:
    """Configuration for each stochastic xApp"""
    name: str
    slice_means: Dict[str, float]  # Mean PRB allocation per slice
    std_dev: float = 1.5


# ============================================================================
# STOCHASTIC XAPP CLASS
# ============================================================================

class StochasticxApp:
    """
    Stochastic xApp that generates slicing policies based on Gaussian distribution.
    
    Core Logic:
    1. Each xApp has a Gaussian distribution with specific mean for each slice
    2. Every 250ms, generates a new random policy by drawing from distribution
    3. Rounds to nearest RBG (Resource Block Group)
    4. Validates total allocation ≤ 50 PRBs
    5. Handles overflow by proportional reduction
    
    Based on PACIFISTA paper Section 9 experiment setup.
    """
    
    def __init__(self, config: xAppConfig, total_prbs: int = 50, rbg_size: int = 2):
        """
        Args:
            config: xAppConfig with name and slice means
            total_prbs: Total available PRBs (default 50)
            rbg_size: Size of each Resource Block Group (default 2)
        """
        self.name = config.name
        self.slice_means = config.slice_means
        self.std_dev = config.std_dev
        self.total_prbs = total_prbs
        self.rbg_size = rbg_size
        
        # History tracking for profiling
        self.policy_history: List[SlicePolicy] = []
        self.last_policy: SlicePolicy | None = None
        
    def generate_policy(self, timestamp: float, rng: np.random.Generator) -> SlicePolicy:
        """
        Generate a new PRB allocation policy.
        
        Process (from PACIFISTA paper):
        1. Draw from Gaussian for eMBB and URLLC
        2. Round to nearest RBG boundary
        3. Validate and adjust allocation
        4. Assign remaining PRBs to mMTC
        
        Args:
            timestamp: Current simulation time (seconds)
            rng: NumPy random generator
            
        Returns:
            SlicePolicy with PRB allocations
        """
        
        # Step 1: Draw from Gaussian distribution for eMBB and URLLC
        embb_draw = rng.normal(
            loc=self.slice_means['eMBB'],
            scale=self.std_dev
        )
        urllc_draw = rng.normal(
            loc=self.slice_means['URLLC'],
            scale=self.std_dev
        )
        
        # Step 2: Round to nearest RBG
        # Each RBG = rbg_size PRBs, so round to nearest multiple
        embb_rbgs = round(embb_draw / self.rbg_size)
        urllc_rbgs = round(urllc_draw / self.rbg_size)
        
        embb_prbs = embb_rbgs * self.rbg_size
        urllc_prbs = urllc_rbgs * self.rbg_size
        
        # Step 3: Clamp to non-negative values
        embb_prbs = max(0, embb_prbs)
        urllc_prbs = max(0, urllc_prbs)
        
        # Step 4: Validate and adjust total allocation
        total_allocated = embb_prbs + urllc_prbs
        
        if total_allocated > self.total_prbs:
            # Proportional reduction to fit within 50 PRBs
            scale_factor = self.total_prbs / total_allocated
            embb_prbs = int(embb_prbs * scale_factor)
            urllc_prbs = int(urllc_prbs * scale_factor)
            
            # Round down to RBG boundaries after scaling
            embb_prbs = (embb_prbs // self.rbg_size) * self.rbg_size
            urllc_prbs = (urllc_prbs // self.rbg_size) * self.rbg_size
        
        # Step 5: Assign remaining PRBs to mMTC
        mmtc_prbs = self.total_prbs - embb_prbs - urllc_prbs
        
        # Create policy
        policy = SlicePolicy(
            eMBB=int(embb_prbs),
            URLLC=int(urllc_prbs),
            mMTC=int(mmtc_prbs),
            timestamp=timestamp
        )
        
        # Store in history
        self.policy_history.append(policy)
        self.last_policy = policy
        
        return policy
    
    def validate_policy(self, policy: SlicePolicy) -> bool:
        """Validate that policy doesn't exceed PRB limits"""
        total = policy.eMBB + policy.URLLC + policy.mMTC
        return (total <= self.total_prbs and 
                policy.eMBB >= 0 and 
                policy.URLLC >= 0 and 
                policy.mMTC >= 0)
    
    def get_statistics(self) -> Dict:
        """
        Compute statistics for profiling.
        
        Returns dictionary with mean, std, min, max for each slice.
        This is part of the statistical profile used in PACIFISTA.
        """
        if not self.policy_history:
            return {}
        
        stats = {}
        for slice_name in ['eMBB', 'URLLC', 'mMTC']:
            values = [getattr(p, slice_name) for p in self.policy_history]
            stats[slice_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': int(np.min(values)),
                'max': int(np.max(values)),
            }
        
        return stats


# ============================================================================
# XAPP CONTROLLER (orchestrates all 5 xApps)
# ============================================================================

class xAppController:
    """
    Manages all 5 stochastic xApps (x1-x5).
    
    Responsibilities:
    - Generate policies for all xApps every 250ms
    - Track policy history for conflict detection
    - Compute statistical profiles for PACIFISTA evaluation
    """
    
    def __init__(self, num_xapps: int = 5, seed: int = 42):
        """
        Initialize controller with 5 xApps (x1 to x5).
        Each has different Gaussian distribution means.
        
        Args:
            num_xapps: Number of xApps (default 5 for PACIFISTA experiment)
            seed: Random seed for reproducibility
        """
        self.num_xapps = num_xapps
        self.xapps: Dict[str, StochasticxApp] = {}
        self.rng = np.random.default_rng(seed)
        
        # Conflict tracking
        self.conflicts: Dict[str, List[Dict]] = defaultdict(list)
        
        # Profiling data (KPMs from RAN environment)
        self.kpm_history: Dict[str, List[Dict]] = defaultdict(list)
        
        self._setup_xapps()
    
    def _setup_xapps(self):
        """
        Setup 5 xApps with different slice mean distributions.
        
        Distribution Strategy (from PACIFISTA paper Section 9):
        - Each xApp has a different "preference" for slice allocation
        - This creates natural conflicts in resource contention
        - Different means simulate different service priorities
        """
        
        # Based on PACIFISTA paper Table 1
        configs = [
            xAppConfig(
                name='x1',
                slice_means={'eMBB': 20, 'URLLC': 15, 'mMTC': 15}
            ),
            xAppConfig(
                name='x2',
                slice_means={'eMBB': 15, 'URLLC': 20, 'mMTC': 15}
            ),
            xAppConfig(
                name='x3',
                slice_means={'eMBB': 18, 'URLLC': 12, 'mMTC': 20}
            ),
            xAppConfig(
                name='x4',
                slice_means={'eMBB': 22, 'URLLC': 18, 'mMTC': 10}
            ),
            xAppConfig(
                name='x5',
                slice_means={'eMBB': 16, 'URLLC': 16, 'mMTC': 18}
            ),
        ]
        
        for config in configs:
            self.xapps[config.name] = StochasticxApp(config)
    
    def generate_policies(self, timestamp: float) -> Dict[str, SlicePolicy]:
        """
        Generate new policies for all xApps at current timestamp.
        
        Called every 250ms in the simulation.
        
        Args:
            timestamp: Current simulation time
            
        Returns:
            Dictionary mapping xApp names to their generated policies
        """
        policies = {}
        for xapp_name, xapp in self.xapps.items():
            policy = xapp.generate_policy(timestamp, self.rng)
            if not xapp.validate_policy(policy):
                print(f"⚠️  INVALID policy from {xapp_name} at t={timestamp:.3f}s")
            policies[xapp_name] = policy
        
        return policies
    
    def get_current_policies(self) -> Dict[str, SlicePolicy]:
        """Get the most recent policy for each xApp"""
        return {name: xapp.last_policy for name, xapp in self.xapps.items() 
                if xapp.last_policy is not None}
    
    def record_kpm(self, xapp_name: str, kpm_data: Dict) -> None:
        """
        Record KPM data for a specific xApp.
        
        KPMs tracked:
        - Throughput (Mbps) per slice
        - Buffer size (bits) per slice
        - Accepted/dropped bits
        
        Args:
            xapp_name: Name of xApp (x1, x2, etc.)
            kpm_data: Dictionary with KPM measurements
        """
        if xapp_name in self.xapps:
            self.kpm_history[xapp_name].append(kpm_data)
    
    def get_profiles(self) -> Dict[str, Dict]:
        """
        Generate statistical profiles for all xApps.
        
        Returns:
            Dictionary mapping xApp names to their profiles containing:
            - Parameter statistics (PRB allocations)
            - KPM statistics (throughput, buffer, etc.)
        """
        profiles = {}
        for xapp_name, xapp in self.xapps.items():
            profiles[xapp_name] = {
                'parameters': xapp.get_statistics(),
                'kpms': self._compute_kpm_stats(xapp_name),
            }
        
        return profiles
    
    def _compute_kpm_stats(self, xapp_name: str) -> Dict:
        """
        Compute KPM statistics from recorded history.
        
        KPMs:
        - Throughput per slice (Mbps)
        - Buffer size per slice (Mbits)
        """
        if xapp_name not in self.kpm_history or not self.kpm_history[xapp_name]:
            return {}
        
        history = self.kpm_history[xapp_name]
        stats = {}
        
        # Aggregate KPMs across all measurements
        kpm_keys = set()
        for kpm in history:
            kpm_keys.update(kpm.keys())
        
        for key in kpm_keys:
            values = [kpm[key] for kpm in history if key in kpm]
            if values:
                stats[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                }
        
        return stats
    
    def detect_pairwise_conflicts(self) -> Dict[Tuple[str, str], float]:
        """
        Detect conflicts between xApps pairwise.
        
        Uses K-S distance to compare ECDFs of policy distributions.
        
        Returns:
            Dictionary mapping (xapp1, xapp2) -> conflict_severity
        """
        from scipy.stats import ks_2samp
        
        conflicts = {}
        xapp_names = sorted(self.xapps.keys())
        
        # Pairwise comparison for all xApp pairs
        for i, name1 in enumerate(xapp_names):
            for name2 in xapp_names[i+1:]:
                xapp1 = self.xapps[name1]
                xapp2 = self.xapps[name2]
                
                # Compare eMBB PRB allocations (main conflict metric)
                vals1 = [p.eMBB for p in xapp1.policy_history]
                vals2 = [p.eMBB for p in xapp2.policy_history]
                
                if len(vals1) > 1 and len(vals2) > 1:
                    ks_stat, _ = ks_2samp(vals1, vals2)
                    conflicts[(name1, name2)] = ks_stat
        
        return conflicts


# ============================================================================
# INTEGRATION WITH RAN ENVIRONMENT
# ============================================================================

class xAppControlledRAN:
    """
    Wrapper around RANSandboxEnv that integrates xApp control.
    
    The xApps provide slicing policies, which the RAN applies.
    KPM measurements from RAN are recorded in xApp profiles.
    """
    
    def __init__(self, env, xapp_controller: xAppController, 
                 policy_update_interval_s: float = 0.25):
        """
        Args:
            env: RANSandboxEnv instance
            xapp_controller: xAppController instance
            policy_update_interval_s: How often to update policy (250ms per paper)
        """
        self.env = env
        self.xapp_controller = xapp_controller
        self.policy_update_interval_s = policy_update_interval_s
        self.next_policy_update_t = policy_update_interval_s
    
    def step(self) -> None:
        """
        Advance simulation by one time step.
        Update xApp policies every 250ms.
        Record KPMs.
        """
        # Update policies every 250ms
        if self.env.t_s >= self.next_policy_update_t:
            policies = self.xapp_controller.generate_policies(self.env.t_s)
            
            # Apply policies to RAN
            for xapp_name, policy in policies.items():
                self.env.set_slicing(policy.eMBB, policy.URLLC)
                # Note: In real scenario, would rotate which xApp's policy is applied
                # For profiling, we run each xApp separately
            
            self.next_policy_update_t += self.policy_update_interval_s
        
        # Step the RAN environment
        self.env.step()
        
        # Record KPMs for current xApp
        # In profiling mode, we simulate one xApp at a time
        snap = self.env.snapshot()
        kpm_data = {
            'time': self.env.t_s,
            'eMBB_throughput_mbps': 0.0,  # Computed from sent_bits
            'eMBB_buffer_mbits': snap['slice_buffer_bits']['eMBB'] / 1e6,
            'URLLC_buffer_mbits': snap['slice_buffer_bits']['URLLC'] / 1e6,
        }
        
        # Note: Would need more detailed tracking for throughput


# ============================================================================
# PROFILING HELPER
# ============================================================================

def profile_single_xapp(env, xapp_name: str, duration_s: float = 60.0, 
                        num_samples_target: int = 3000) -> Dict:
    """
    Profile a single xApp in isolation.
    
    Procedure (from PACIFISTA Section 5.1):
    1. Run xApp in sandbox environment
    2. Collect at least 3,000 samples per slice
    3. Compute ECDFs and statistics
    4. Generate application profile
    
    Args:
        env: RANSandboxEnv instance
        xapp_name: Name of xApp to profile (x1, x2, etc.)
        duration_s: How long to run (default 60 seconds)
        num_samples_target: Target number of samples (default 3000)
        
    Returns:
        Statistical profile dictionary
    """
    from scipy import stats as sp_stats
    
    # Create controller with single xApp for testing
    controller = xAppController(num_xapps=1, seed=42)
    xapp = controller.xapps[xapp_name]
    
    # Run simulation
    env.tx_on = True
    policy_times = []
    
    while env.t_s < duration_s:
        # Generate policy every 250ms
        if len(policy_times) == 0 or env.t_s >= policy_times[-1] + 0.25:
            policy = xapp.generate_policy(env.t_s, controller.rng)
            env.set_slicing(policy.eMBB, policy.URLLC)
            policy_times.append(env.t_s)
        
        env.step()
        
        # Early exit if we have enough samples
        if len(xapp.policy_history) >= num_samples_target:
            break
    
    # Compute ECDFs and statistics
    profile = {
        'xapp_name': xapp_name,
        'duration_s': env.t_s,
        'num_samples': len(xapp.policy_history),
        'parameters': xapp.get_statistics(),
    }
    
    # Compute ECDFs for parameters
    for slice_name in ['eMBB', 'URLLC', 'mMTC']:
        values = [getattr(p, slice_name) for p in xapp.policy_history]
        
        # Empirical CDF
        sorted_vals = np.sort(values)
        ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        profile[f'{slice_name}_ecdf'] = {
            'x': sorted_vals.tolist(),
            'y': ecdf_y.tolist(),
        }
    
    return profile


def compare_xapps_pairwise(profile1: Dict, profile2: Dict) -> Dict:
    """
    Compare two xApp profiles using K-S distance.
    
    Computes conflict severity between a pair of xApps.
    
    Args:
        profile1, profile2: Profile dictionaries from profile_single_xapp()
        
    Returns:
        Conflict report with K-S distances
    """
    from scipy.stats import ks_2samp
    
    report = {
        'xapp1': profile1['xapp_name'],
        'xapp2': profile2['xapp_name'],
        'ks_distances': {},
    }
    
    # Compare each slice parameter distribution
    for slice_name in ['eMBB', 'URLLC', 'mMTC']:
        ecdf1 = profile1[f'{slice_name}_ecdf']
        ecdf2 = profile2[f'{slice_name}_ecdf']
        
        # Use original policy values for K-S test
        vals1 = profile1['parameters'][slice_name]
        vals2 = profile2['parameters'][slice_name]
        
        # Approximate with mean ± std (for simplified comparison)
        # In real PACIFISTA, would use full ECDFs
        ks_stat, p_value = ks_2samp(
            np.random.normal(vals1['mean'], vals1['std'], 100),
            np.random.normal(vals2['mean'], vals2['std'], 100),
        )
        
        report['ks_distances'][slice_name] = {
            'statistic': float(ks_stat),
            'p_value': float(p_value),
        }
    
    return report


if __name__ == "__main__":
    print("Stochastic xApps module for PACIFISTA conflict evaluation.")
    print("Import this module to use in main.py or profiling scripts.")