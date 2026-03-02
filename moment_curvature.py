"""
============================================================
Moment-Curvature Analysis of a Confined RC Section
Modified Kent & Park Model
============================================================
Author: Bhoshaga Mitrran Ravi Chandran
Website: https://bhosh.com
Interactive Web App: https://bhosh.com/moment-curvature

Description:
Computes the M-phi relationship for a rectangular RC section
using Kent & Park (1971) and Modified Kent & Park (1982)
concrete models. 14"x20" section, 8 #9 bars, #4 ties @ 6",
f'c = 4000 psi, fy = 60 ksi.

Modes:
  1. whole - Confined model for entire section
  2. split - Unconfined cover + confined core
  3. full  - Split + steel strain hardening

Usage:
  python moment_curvature.py                # runs all three modes
  python moment_curvature.py --mode whole   # single mode
  python moment_curvature.py --mode all --no-plot  # headless
============================================================
"""

import argparse
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# --- Section Geometry (inches) ---
b = 14.0          # section width
h = 20.0          # section depth
cover_bar = 3.0   # distance from edge to center of bars

# Steel layers: (depth from top in inches, number of bars)
# #9 bars: A_bar = 1.00 in^2
A_bar = 1.00
steel_layers = [
    (3.0,  3),   # 3 #9 bars at 3" from top
    (10.0, 2),   # 2 #9 bars at 10" from top (mid-height)
    (17.0, 3),   # 3 #9 bars at 17" from top
]

# Ties: #4 bars
d_tie = 0.50      # #4 bar diameter (in)
A_tie = 0.20      # #4 bar area (in^2)
s_h = 6.0         # tie spacing (in)

# Clear cover to outside of ties: 3 - 0.564 - 0.5 ~ 2"
clear_cover = 2.0

# Confined core (to outside of hoops)
b_core = b - 2 * clear_cover   # 10.0 in
d_core = h - 2 * clear_cover   # 16.0 in
y_core_top = clear_cover        # 2.0 in from top
y_core_bot = h - clear_cover    # 18.0 in from top

# --- Material Properties ---
fc_prime = 4000.0   # psi (unconfined compressive strength)
fy = 60.0           # ksi (steel yield strength)
Es = 29000.0        # ksi (steel modulus)
f_yh = 60.0         # ksi (tie yield strength)

fy_psi = fy * 1000
f_yh_psi = f_yh * 1000
eps_y = fy / Es           # 0.00207

# Strain hardening parameters
eps_sh = 0.008     # strain at onset of hardening
f_su = 90.0        # ksi, ultimate stress
eps_su = 0.10      # ultimate strain

# --- Confinement Parameters ---
rho_s = 2 * (b_core + d_core) * A_tie / (b_core * d_core * s_h)
K = 1 + rho_s * f_yh_psi / fc_prime

# --- Unconfined Concrete: Kent & Park (1971) ---
eps0_u = 0.002
eps_50u = (3 + 0.002 * fc_prime) / (fc_prime - 1000)
Z_u = 0.5 / (eps_50u - eps0_u)
eps_20u = eps0_u + 0.8 / Z_u

# --- Confined Concrete: Modified Kent & Park (1982) ---
eps0_c = 0.002 * K
eps_50h = 0.75 * rho_s * np.sqrt(b_core / s_h)
Z_m = 0.5 / (eps_50u + eps_50h - 0.002 * K)
eps_20c = 0.002 * K + 0.8 / Z_m


# --- Material Models ---
def concrete_stress_unconfined(eps_c):
    """Unconfined concrete stress (psi). Kent & Park 1971.
    Returns 0 after cover spalling (eps > eps_20u)."""
    if eps_c <= 0:
        return 0.0
    if eps_c <= eps0_u:
        return fc_prime * (2 * eps_c / eps0_u - (eps_c / eps0_u)**2)
    elif eps_c <= eps_20u:
        stress = fc_prime * (1 - Z_u * (eps_c - eps0_u))
        return max(stress, 0.2 * fc_prime)
    else:
        return 0.0  # cover spalls off


def concrete_stress_confined(eps_c):
    """Confined concrete stress (psi). Modified Kent & Park 1982."""
    if eps_c <= 0:
        return 0.0
    if eps_c <= eps0_c:
        return K * fc_prime * (2 * eps_c / eps0_c - (eps_c / eps0_c)**2)
    elif eps_c <= eps_20c:
        stress = K * fc_prime * (1 - Z_m * (eps_c - eps0_c))
        return max(stress, 0.2 * K * fc_prime)
    else:
        return 0.2 * K * fc_prime


def steel_stress_ksi(eps_s, strain_hardening=False):
    """Steel stress in ksi. Positive = tension, negative = compression."""
    if not strain_hardening:
        if abs(eps_s) <= eps_y:
            return Es * eps_s
        else:
            return fy * np.sign(eps_s)
    else:
        if abs(eps_s) <= eps_y:
            return Es * eps_s
        elif abs(eps_s) <= eps_sh:
            return fy * np.sign(eps_s)
        elif abs(eps_s) <= eps_su:
            f = fy + (f_su - fy) * (abs(eps_s) - eps_sh) / (eps_su - eps_sh)
            return f * np.sign(eps_s)
        else:
            return f_su * np.sign(eps_s)


# --- M-phi Analysis ---
n_strips = 1000
strip_h = h / n_strips
strip_y = np.array([strip_h * (i + 0.5) for i in range(n_strips)])


def compute_section_response(eps_cm, c, mode="whole"):
    """Net axial force (kips) and moment (kip-in) about mid-depth."""
    centroid = h / 2.0
    use_sh = (mode == "full")
    total_force = 0.0
    total_moment = 0.0

    # Concrete strips
    for i in range(n_strips):
        y = strip_y[i]
        eps_c = eps_cm * (c - y) / c

        if mode == "whole":
            stress_psi = concrete_stress_confined(eps_c)
        else:
            if y_core_top <= y <= y_core_bot:
                stress_psi = concrete_stress_confined(eps_c)
            else:
                stress_psi = concrete_stress_unconfined(eps_c)

        force = stress_psi * b * strip_h / 1000.0
        lever = centroid - y
        total_force += force
        total_moment += force * lever

    # Steel layers
    for (d_i, n_bars) in steel_layers:
        eps_s = eps_cm * (c - d_i) / c
        f_s = steel_stress_ksi(-eps_s, strain_hardening=use_sh)
        A_s = n_bars * A_bar
        force_steel_kips = -f_s * A_s

        # Subtract displaced concrete
        if eps_s > 0:
            if mode == "whole":
                conc_stress = concrete_stress_confined(eps_s)
            else:
                if y_core_top <= d_i <= y_core_bot:
                    conc_stress = concrete_stress_confined(eps_s)
                else:
                    conc_stress = concrete_stress_unconfined(eps_s)
            displaced_conc_force = conc_stress * A_s / 1000.0
        else:
            displaced_conc_force = 0.0

        net_steel_force = force_steel_kips - displaced_conc_force
        lever = centroid - d_i
        total_force += net_steel_force
        total_moment += net_steel_force * lever

    return total_force, total_moment


def find_neutral_axis(eps_cm, mode="whole"):
    """Find neutral axis depth c where net force = 0 (bisection via brentq)."""
    c_low, c_high = 0.1, h * 3

    f_low = compute_section_response(eps_cm, c_low, mode)[0]
    f_high = compute_section_response(eps_cm, c_high, mode)[0]

    if f_low * f_high > 0:
        c_low, c_high = 0.01, h * 5
        f_low = compute_section_response(eps_cm, c_low, mode)[0]
        f_high = compute_section_response(eps_cm, c_high, mode)[0]
        if f_low * f_high > 0:
            return None

    try:
        return brentq(lambda c: compute_section_response(eps_cm, c, mode)[0],
                      c_low, c_high, xtol=1e-6)
    except ValueError:
        return None


def run_moment_curvature(mode="whole"):
    """Run M-phi analysis for given mode. Returns (eps, phi, M, c) arrays."""
    eps_max = 1.5 * eps_20c
    eps_values = np.arange(0.0002, eps_max + 0.0001, 0.0002)

    phi_list, M_list, c_list, eps_list = [], [], [], []

    for eps_cm in eps_values:
        c = find_neutral_axis(eps_cm, mode)
        if c is None:
            continue
        _, M = compute_section_response(eps_cm, c, mode)
        phi_list.append(eps_cm / c)
        M_list.append(M / 12.0)
        c_list.append(c)
        eps_list.append(eps_cm)

    return (np.array(eps_list), np.array(phi_list),
            np.array(M_list), np.array(c_list))


# --- Output helpers ---
MODE_NAMES = {
    "whole": "Confined for whole section, no strain hardening",
    "split": "Unconfined cover + confined core, no strain hardening",
    "full":  "Unconfined cover + confined core, with strain hardening",
}


def print_parameters():
    print("=" * 60)
    print("CE 676 - Assignment 3: Moment-Curvature Analysis")
    print("Modified Kent & Park Model")
    print("=" * 60)

    print(f"\nSection: {b:.0f}\" x {h:.0f}\"")
    print(f"Steel: 8 #9 bars (A_bar = {A_bar:.2f} in^2 each)")
    for i, (d_i, n_b) in enumerate(steel_layers):
        print(f"  Layer {i+1}: {n_b} bars at {d_i:.1f}\" from top")
    print(f"Ties: #4 @ {s_h:.0f}\" (A_tie = {A_tie:.2f} in^2)")
    print(f"Confined core: b\" = {b_core:.1f}\", d\" = {d_core:.1f}\"")

    print(f"\nf'c = {fc_prime:.0f} psi, fy = {fy:.0f} ksi, Es = {Es:.0f} ksi")
    print(f"eps_y = {eps_y:.6f}")
    print(f"rho_s = {rho_s:.6f}, K = {K:.4f}")

    print(f"\nUnconfined: eps_0 = {eps0_u}, eps_50u = {eps_50u:.6f}, "
          f"Z_u = {Z_u:.2f}, eps_20,u = {eps_20u:.5f}")
    print(f"Confined:   eps_0c = {eps0_c:.6f}, eps_50h = {eps_50h:.6f}, "
          f"Z_m = {Z_m:.2f}, eps_20,c = {eps_20c:.5f}")
    print(f"Peak: Kf'c = {K*fc_prime:.0f} psi = {K*fc_prime/1000:.3f} ksi, "
          f"Residual: 0.2Kf'c = {0.2*K*fc_prime:.0f} psi")


def print_results_table(eps_arr, phi_arr, M_arr, c_arr, mode):
    key_strains = [0.001, 0.002, 0.003, 0.004, 0.005, eps_20c, 1.5 * eps_20c]
    key_labels = ["0.001", "0.002", "0.003", "0.004", "0.005",
                  f"eps_20c={eps_20c:.5f}", f"1.5*eps_20c={1.5*eps_20c:.5f}"]

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {MODE_NAMES[mode]}")
    print(f"{'=' * 70}")
    print(f"{'eps_cm':>20s}  {'c (in)':>8s}  {'phi (1/in)':>12s}  {'M (kip-ft)':>12s}")
    print("-" * 70)

    for strain_val, label in zip(key_strains, key_labels):
        idx = np.argmin(np.abs(eps_arr - strain_val))
        if abs(eps_arr[idx] - strain_val) < 0.0003:
            print(f"{label:>20s}  {c_arr[idx]:8.3f}  "
                  f"{phi_arr[idx]:12.6f}  {M_arr[idx]:12.2f}")
        else:
            print(f"{label:>20s}  {'N/A':>8s}  {'N/A':>12s}  {'N/A':>12s}")


def plot_stress_strain(out_dir):
    """Plot concrete and steel stress-strain curves with region annotations."""
    eps_plot = np.linspace(0, 0.04, 2000)
    stress_unconf = np.array([concrete_stress_unconfined(e) for e in eps_plot])
    stress_conf = np.array([concrete_stress_confined(e) for e in eps_plot])

    # --- Concrete stress-strain ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(eps_plot, stress_unconf / 1000, 'b-', linewidth=2,
             label='Unconfined (Kent & Park 1971)')
    ax1.plot(eps_plot, stress_conf / 1000, 'r--', linewidth=2,
             label='Confined (Mod. Kent & Park 1982)')

    # Annotate regions on confined curve (B, C, D)
    mid_B = eps0_c / 2
    mid_C = (eps0_c + eps_20c) / 2
    mid_D = eps_20c + 0.004
    peak_ksi = K * fc_prime / 1000
    res_ksi = 0.2 * K * fc_prime / 1000
    ax1.annotate('B', xy=(mid_B, peak_ksi * 0.6), fontsize=14,
                 fontweight='bold', color='red', ha='center')
    ax1.annotate('C', xy=(mid_C, (peak_ksi + res_ksi) / 2 + 0.3), fontsize=14,
                 fontweight='bold', color='red', ha='center')
    ax1.annotate('D', xy=(mid_D, res_ksi + 0.25), fontsize=14,
                 fontweight='bold', color='red', ha='center')

    ax1.axvline(x=eps_20u, color='b', linestyle=':', alpha=0.5,
                label=f'$\\varepsilon_{{20,u}}$ = {eps_20u:.4f}')
    ax1.axvline(x=eps_20c, color='r', linestyle=':', alpha=0.5,
                label=f'$\\varepsilon_{{20,c}}$ = {eps_20c:.4f}')

    ax1.plot(eps0_u, fc_prime / 1000, 'bo', markersize=5)
    ax1.plot(eps0_c, peak_ksi, 'rs', markersize=5)

    ax1.set_xlabel('Strain, $\\varepsilon_c$', fontsize=13)
    ax1.set_ylabel('Stress, $f_c$ (ksi)', fontsize=13)
    ax1.set_title('Concrete Stress-Strain Curves (KLP Model)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.set_xlim([0, 0.035])
    ax1.set_ylim([0, max(peak_ksi * 1.15, 5.5)])
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/stress_strain.png', dpi=300, bbox_inches='tight')
    print("Plot saved: src/stress_strain.png")

    # --- Steel stress-strain ---
    eps_s_plot = np.linspace(-0.05, 0.05, 2000)
    stress_epp = np.array([steel_stress_ksi(e, strain_hardening=False)
                           for e in eps_s_plot])
    stress_sh = np.array([steel_stress_ksi(e, strain_hardening=True)
                          for e in eps_s_plot])

    fig_s, ax_s = plt.subplots(figsize=(10, 5))
    ax_s.plot(eps_s_plot, stress_epp, 'b-', linewidth=2,
              label='Elastic-Perfectly Plastic')
    ax_s.plot(eps_s_plot, stress_sh, 'r--', linewidth=2,
              label=f'With Strain Hardening ($f_{{su}}$ = {f_su:.0f} ksi)')
    ax_s.axhline(y=fy, color='gray', linestyle=':', alpha=0.4)
    ax_s.axhline(y=-fy, color='gray', linestyle=':', alpha=0.4)
    ax_s.axvline(x=eps_y, color='gray', linestyle=':', alpha=0.4)
    ax_s.axvline(x=-eps_y, color='gray', linestyle=':', alpha=0.4)
    ax_s.set_xlabel('Strain, $\\varepsilon_s$', fontsize=13)
    ax_s.set_ylabel('Stress, $f_s$ (ksi)', fontsize=13)
    ax_s.set_title('Steel Stress-Strain Model', fontsize=14)
    ax_s.legend(fontsize=11)
    ax_s.set_xlim([-0.05, 0.05])
    ax_s.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/steel_stress_strain.png', dpi=300, bbox_inches='tight')
    print("Plot saved: src/steel_stress_strain.png")

    # Export CSV for pgfplots
    idx_ss = np.linspace(0, len(eps_plot) - 1, 500, dtype=int)
    with open(f'{out_dir}/stress_strain.csv', 'w') as f:
        f.write('strain,unconfined_ksi,confined_ksi\n')
        for i in idx_ss:
            f.write(f'{eps_plot[i]:.6f},{stress_unconf[i]/1000:.4f},'
                    f'{stress_conf[i]/1000:.4f}\n')

    return fig1, fig_s


def plot_moment_curvature(results_by_mode, out_dir):
    """Plot M-phi curves. results_by_mode = {mode: (eps, phi, M, c), ...}"""
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    colors = {"whole": "k", "split": "b", "full": "r"}
    styles = {"whole": "-", "split": "--", "full": "-."}

    for mode, (eps_arr, phi_arr, M_arr, c_arr) in results_by_mode.items():
        ax2.plot(phi_arr, M_arr, color=colors[mode], linestyle=styles[mode],
                 linewidth=2, label=MODE_NAMES[mode])

    # Mark key points on the first (primary) mode
    primary_mode = list(results_by_mode.keys())[0]
    eps_arr, phi_arr, M_arr, c_arr = results_by_mode[primary_mode]

    key_strains = [0.001, 0.002, 0.003, 0.004, 0.005, eps_20c]
    key_labels = ["0.001", "0.002", "0.003", "0.004", "0.005",
                  "$\\varepsilon_{20,c}$"]

    for strain_val, label in zip(key_strains, key_labels):
        idx = np.argmin(np.abs(eps_arr - strain_val))
        if abs(eps_arr[idx] - strain_val) < 0.0003:
            ax2.plot(phi_arr[idx], M_arr[idx], 'ro', markersize=5)
            ax2.annotate(label, (phi_arr[idx], M_arr[idx]),
                         textcoords="offset points", xytext=(8, 5), fontsize=7)

    ax2.set_xlabel('Curvature, $\\phi$ (1/in)', fontsize=13)
    ax2.set_ylabel('Moment, M (kip-ft)', fontsize=13)
    ax2.set_title('Moment-Curvature Relationship', fontsize=14)
    if len(results_by_mode) > 1:
        ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/moment_curvature.png', dpi=300, bbox_inches='tight')
    print("Plot saved: src/moment_curvature.png")

    # Export CSV per mode for pgfplots
    for mode, (e_arr, p_arr, m_arr, ca_arr) in results_by_mode.items():
        with open(f'{out_dir}/mphi_{mode}.csv', 'w') as f:
            f.write('phi,moment_kipft,eps_cm,c_in\n')
            for i in range(len(p_arr)):
                f.write(f'{p_arr[i]:.8f},{m_arr[i]:.4f},'
                        f'{e_arr[i]:.6f},{ca_arr[i]:.4f}\n')

    # Key points CSV (primary mode)
    with open(f'{out_dir}/key_points.csv', 'w') as f:
        f.write('label,phi,moment_kipft,eps_cm\n')
        all_key = [0.001, 0.002, 0.003, 0.004, 0.005, eps_20c, 1.5 * eps_20c]
        all_lbl = ["0.001", "0.002", "0.003", "0.004", "0.005",
                   "eps_20c", "1.5*eps_20c"]
        for sv, lb in zip(all_key, all_lbl):
            idx = np.argmin(np.abs(eps_arr - sv))
            if abs(eps_arr[idx] - sv) < 0.0003:
                f.write(f'{lb},{phi_arr[idx]:.8f},{M_arr[idx]:.4f},'
                        f'{eps_arr[idx]:.6f}\n')

    return fig2


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CE 676 HW3 - Moment-Curvature Analysis")
    parser.add_argument(
        "--mode", choices=["whole", "split", "full", "all"],
        default="all",
        help="Analysis mode: whole, split, full, or all (default)")
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip matplotlib display (still saves PNGs)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    os.makedirs(out_dir, exist_ok=True)

    # Print parameters
    print_parameters()

    # Determine which modes to run
    if args.mode == "all":
        modes = ["whole", "split", "full"]
    else:
        modes = [args.mode]

    # Run analysis for each mode
    results = {}
    for mode in modes:
        print(f"\nRunning: {MODE_NAMES[mode]}")
        print("-" * 60)
        eps_arr, phi_arr, M_arr, c_arr = run_moment_curvature(mode)
        results[mode] = (eps_arr, phi_arr, M_arr, c_arr)
        print_results_table(eps_arr, phi_arr, M_arr, c_arr, mode)
        print(f"  {len(eps_arr)} converged points")

    # Plots
    plot_stress_strain(out_dir)
    plot_moment_curvature(results, out_dir)
    print("\nCSV files saved to src/")

    if not args.no_plot:
        plt.show()

    print("Done.")
