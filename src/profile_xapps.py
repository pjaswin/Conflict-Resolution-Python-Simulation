# profile_xapps.py
"""
Profiling script to collect statistical profiles for all 5 xApps.

Based on PACIFISTA Section 5.1 profiling methodology:
1. Run each xApp in sandbox environment
2. Collect at least 3,000 samples per slice
3. Compute ECDFs and statistics
4. Generate application profiles

Usage:
  python profile_xapps.py

Output:
  profiles/x1_profile.json
  profiles/x2_profile.json
  ...
  conflict_matrix.csv (pairwise K-S distances)
"""

from __future__ import annotations

import json
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

# Import your modules
from ran_environment import SimConfig, RANSandboxEnv
from stochastic_xapps import xAppController, profile_single_xapp, compare_xapps_pairwise


# ============================================================================
# PROFILING CONFIGURATION
# ============================================================================

PROFILING_CONFIG = {
    'num_samples_per_xapp': 3000,      # Minimum samples per xApp
    'duration_per_xapp_s': 300.0,      # 5 minutes per xApp (plenty of time to get 3000 samples)
    'xapp_names': ['x1', 'x2', 'x3', 'x4', 'x5'],
    'output_dir': 'profiles',
}


# ============================================================================
# PROFILING FUNCTION
# ============================================================================

def profile_all_xapps(config: PROFILING_CONFIG | None = None):
    """
    Profile all 5 xApps in isolation.
    
    Procedure:
    1. For each xApp, create a fresh RAN environment
    2. Run xApp for fixed duration or until 3000 samples collected
    3. Compute statistical profile (mean, std, ECDF)
    4. Save profile to JSON
    
    Args:
        config: Configuration dictionary
    """
    if config is None:
        config = PROFILING_CONFIG
    
    profiles = {}
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("\n" + "="*80)
    print("PACIFISTA xApp PROFILING")
    print("="*80)
    print(f"Target samples per xApp: {config['num_samples_per_xapp']}")
    print(f"Duration per xApp: {config['duration_per_xapp_s']:.1f}s")
    print(f"Output directory: {config['output_dir']}")
    print("="*80 + "\n")
    
    for xapp_name in config['xapp_names']:
        print(f"\n{'─'*80}")
        print(f"Profiling {xapp_name}...")
        print(f"{'─'*80}")
        
        # Create fresh RAN environment
        ran_cfg = SimConfig()
        env = RANSandboxEnv(ran_cfg)
        
        # Create xApp controller
        controller = xAppController(num_xapps=5, seed=42)
        xapp = controller.xapps[xapp_name]
        
        # Enable transmission
        env.tx_on = True
        
        # Run profiling
        sample_count = 0
        policy_times = []
        last_report_t = 0.0
        
        while env.t_s < config['duration_per_xapp_s']:
            # Generate policy every 250ms
            if len(policy_times) == 0 or env.t_s >= policy_times[-1] + 0.25:
                policy = xapp.generate_policy(env.t_s, controller.rng)
                env.set_slicing(policy.eMBB, policy.URLLC)
                policy_times.append(env.t_s)
                sample_count += 1
            
            # Step the environment
            env.step()
            
            # Progress report every 10 seconds
            if env.t_s - last_report_t >= 10.0:
                print(f"  t={env.t_s:.1f}s | samples={sample_count} | "
                      f"latest policy: eMBB={policy.eMBB} URLLC={policy.URLLC} mMTC={policy.mMTC}")
                last_report_t = env.t_s
            
            # Early exit if we have enough samples
            if sample_count >= config['num_samples_per_xapp']:
                print(f"  Reached {sample_count} samples at t={env.t_s:.1f}s")
                break
        
        # Compute profile
        profile = _compute_profile(xapp, env.t_s)
        profiles[xapp_name] = profile
        
        # Save profile
        profile_path = os.path.join(config['output_dir'], f'{xapp_name}_profile.json')
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        print(f"  ✓ Saved to {profile_path}")
        print(f"\n  Profile Summary:")
        print(f"    Duration: {profile['duration_s']:.1f}s")
        print(f"    Samples: {profile['num_samples']}")
        for slice_name in ['eMBB', 'URLLC', 'mMTC']:
            stats = profile['parameters'][slice_name]
            print(f"    {slice_name}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, "
                  f"range=[{stats['min']}, {stats['max']}]")
    
    return profiles


def _compute_profile(xapp, duration_s: float) -> Dict:
    """
    Compute statistical profile for an xApp.
    
    Includes:
    - Parameter statistics (mean, std, min, max for PRB allocations)
    - ECDFs (Empirical Cumulative Distribution Functions)
    
    Args:
        xapp: StochasticxApp instance after profiling
        duration_s: Total simulation duration
        
    Returns:
        Profile dictionary
    """
    profile = {
        'xapp_name': xapp.name,
        'duration_s': float(duration_s),
        'num_samples': len(xapp.policy_history),
        'parameters': xapp.get_statistics(),
        'ecdf': {},
    }
    
    # Compute ECDFs for each slice parameter
    for slice_name in ['eMBB', 'URLLC', 'mMTC']:
        values = [getattr(p, slice_name) for p in xapp.policy_history]
        
        # Sort for ECDF
        sorted_vals = np.sort(values)
        ecdf_y = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        
        profile['ecdf'][slice_name] = {
            'x': sorted_vals.tolist(),  # PRB values
            'y': ecdf_y.tolist(),        # CDF values
        }
    
    return profile


# ============================================================================
# CONFLICT DETECTION (PAIRWISE)
# ============================================================================

def detect_conflicts_pairwise(profiles: Dict[str, Dict]) -> Dict[Tuple[str, str], Dict]:
    """
    Detect conflicts between all pairs of xApps.
    
    Uses K-S (Kolmogorov-Smirnov) distance to compare ECDFs.
    Computes for each slice (eMBB, URLLC, mMTC).
    
    From PACIFISTA Section 7 - Conflict Evaluation:
    - K-S distance measures maximum vertical distance between ECDFs
    - INT distance is area between ECDFs (not computed here but could be)
    - Higher values = more severe conflict
    
    Args:
        profiles: Dictionary mapping xApp names to their profiles
        
    Returns:
        Dictionary mapping (xapp1, xapp2) pairs to conflict reports
    """
    xapp_names = sorted(profiles.keys())
    conflicts = {}
    
    print("\n" + "="*80)
    print("PAIRWISE CONFLICT DETECTION")
    print("="*80 + "\n")
    
    for i, name1 in enumerate(xapp_names):
        for name2 in xapp_names[i+1:]:
            profile1 = profiles[name1]
            profile2 = profiles[name2]
            
            # Compare parameters using K-S distance
            conflict_report = {
                'xapp1': name1,
                'xapp2': name2,
                'ks_distances': {},
                'severity': 'UNKNOWN',
            }
            
            # Compute K-S distance for each slice
            ks_max = 0.0
            for slice_name in ['eMBB', 'URLLC', 'mMTC']:
                ecdf1 = profile1['ecdf'][slice_name]
                ecdf2 = profile2['ecdf'][slice_name]
                
                # Reconstruct ECDFs for K-S test
                vals1 = ecdf1['x']
                vals2 = ecdf2['x']
                
                ks_stat, p_val = ks_2samp(vals1, vals2)
                
                conflict_report['ks_distances'][slice_name] = {
                    'statistic': float(ks_stat),
                    'p_value': float(p_val),
                }
                
                ks_max = max(ks_max, ks_stat)
            
            # Classify severity (PACIFISTA uses thresholds)
            if ks_max >= 0.5:
                conflict_report['severity'] = 'HIGH'
            elif ks_max >= 0.3:
                conflict_report['severity'] = 'MEDIUM'
            elif ks_max >= 0.1:
                conflict_report['severity'] = 'LOW'
            else:
                conflict_report['severity'] = 'NONE'
            
            conflicts[(name1, name2)] = conflict_report
            
            print(f"{name1} vs {name2}:")
            for slice_name in ['eMBB', 'URLLC', 'mMTC']:
                ks = conflict_report['ks_distances'][slice_name]['statistic']
                print(f"  {slice_name:6s}: K-S={ks:.4f}", end="")
            print(f"  →  {conflict_report['severity']}")
    
    return conflicts


# ============================================================================
# REPORT GENERATION
# ============================================================================

def save_conflict_matrix(conflicts: Dict, output_path: str = 'profiles/conflict_matrix.csv'):
    """
    Save conflict severity matrix to CSV.
    
    Format: xApp pairs vs conflict severity (K-S distance for eMBB)
    
    Args:
        conflicts: Dictionary from detect_conflicts_pairwise()
        output_path: Where to save CSV
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['xApp1', 'xApp2', 'eMBB_KS', 'URLLC_KS', 'mMTC_KS', 'Severity'])
        
        # Data rows
        for (name1, name2), report in sorted(conflicts.items()):
            row = [
                name1,
                name2,
                f"{report['ks_distances']['eMBB']['statistic']:.4f}",
                f"{report['ks_distances']['URLLC']['statistic']:.4f}",
                f"{report['ks_distances']['mMTC']['statistic']:.4f}",
                report['severity'],
            ]
            writer.writerow(row)
    
    print(f"\n✓ Saved conflict matrix to {output_path}")


def generate_profiling_report(profiles: Dict, conflicts: Dict, output_path: str = 'profiles/profiling_report.txt'):
    """
    Generate human-readable profiling report.
    
    Args:
        profiles: Profiles dictionary
        conflicts: Conflicts dictionary
        output_path: Where to save report
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("PACIFISTA PROFILING REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Individual profiles
        f.write("INDIVIDUAL xAPP PROFILES\n")
        f.write("-"*80 + "\n\n")
        
        for xapp_name in sorted(profiles.keys()):
            profile = profiles[xapp_name]
            f.write(f"\n{xapp_name.upper()}\n")
            f.write(f"  Duration: {profile['duration_s']:.1f}s\n")
            f.write(f"  Samples: {profile['num_samples']}\n")
            f.write(f"  Parameters:\n")
            
            for slice_name in ['eMBB', 'URLLC', 'mMTC']:
                stats = profile['parameters'][slice_name]
                f.write(f"    {slice_name:6s}: mean={stats['mean']:6.2f}, std={stats['std']:5.2f}, "
                       f"min={stats['min']:3d}, max={stats['max']:3d}\n")
        
        # Conflict matrix
        f.write("\n" + "="*80 + "\n")
        f.write("PAIRWISE CONFLICT DETECTION\n")
        f.write("-"*80 + "\n\n")
        
        for (name1, name2), report in sorted(conflicts.items()):
            f.write(f"\n{name1} vs {name2}: {report['severity']}\n")
            for slice_name in ['eMBB', 'URLLC', 'mMTC']:
                ks = report['ks_distances'][slice_name]['statistic']
                pval = report['ks_distances'][slice_name]['p_value']
                f.write(f"  {slice_name}: K-S={ks:.4f}, p-value={pval:.4f}\n")
    
    print(f"✓ Saved profiling report to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Run complete profiling workflow:
    1. Profile all 5 xApps
    2. Detect pairwise conflicts
    3. Generate reports
    """
    print("\n" + "="*80)
    print("PACIFISTA xApp PROFILING WORKFLOW")
    print("="*80)
    
    # Step 1: Profile all xApps
    profiles = profile_all_xapps(PROFILING_CONFIG)
    
    # Step 2: Detect conflicts
    conflicts = detect_conflicts_pairwise(profiles)
    
    # Step 3: Save results
    output_dir = PROFILING_CONFIG['output_dir']
    
    save_conflict_matrix(conflicts, os.path.join(output_dir, 'conflict_matrix.csv'))
    generate_profiling_report(profiles, conflicts, os.path.join(output_dir, 'profiling_report.txt'))
    
    print("\n" + "="*80)
    print("PROFILING COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {output_dir}/x1_profile.json through x5_profile.json")
    print(f"  - {output_dir}/conflict_matrix.csv")
    print(f"  - {output_dir}/profiling_report.txt")
    print("\nNext steps:")
    print("  1. Review conflict_matrix.csv for conflict pairs")
    print("  2. Run live conflict evaluation with main.py")
    print("  3. Implement conflict mitigation strategies")


if __name__ == "__main__":
    main()