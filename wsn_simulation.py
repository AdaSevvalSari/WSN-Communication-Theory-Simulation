"""
LEEN354 Communication Theory II — Project Simulation
Topic: D-10 Wireless Sensor Networks
Wireless Sensor Networks: Modulation Performance and Energy-Reliability Trade-offs

Generates all figures for the project report.
Run: python wsn_simulation.py
"""

import numpy as np
from scipy.special import erfc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Output directory ──────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.2,
})

COLORS = {
    "bpsk":   "#1f77b4",
    "qpsk":   "#ff7f0e",
    "oqpsk":  "#2ca02c",
    "fsk":    "#d62728",
    "theory": "#9467bd",
    "actual": "#8c564b",
    "indoor": "#e377c2",
    "outdoor":"#7f7f7f",
}

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — BER vs Eb/N0 for BPSK, QPSK, O-QPSK
# ═══════════════════════════════════════════════════════════════════════════════

def ber_bpsk(ebn0_linear):
    return 0.5 * erfc(np.sqrt(ebn0_linear))

def ber_qpsk(ebn0_linear):
    # Exact expression for QPSK BER
    x = np.sqrt(ebn0_linear)
    return erfc(x) - (0.25) * erfc(x)**2

def ber_oqpsk(ebn0_linear):
    # O-QPSK has the same theoretical BER as QPSK in AWGN
    return ber_qpsk(ebn0_linear)

def ber_fsk_noncoherent(ebn0_linear):
    # Noncoherent FSK: Pe = 0.5 * exp(-Eb/2N0)
    return 0.5 * np.exp(-ebn0_linear / 2)

def plot_ber_vs_ebno():
    ebn0_db  = np.linspace(-4, 14, 500)
    ebn0_lin = 10 ** (ebn0_db / 10)

    ber_b  = ber_bpsk(ebn0_lin)
    ber_q  = ber_qpsk(ebn0_lin)
    ber_oq = ber_oqpsk(ebn0_lin)
    ber_f  = ber_fsk_noncoherent(ebn0_lin)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogy(ebn0_db, ber_b,  color=COLORS["bpsk"],   label="BPSK")
    ax.semilogy(ebn0_db, ber_q,  color=COLORS["qpsk"],   label="QPSK", linestyle="--")
    ax.semilogy(ebn0_db, ber_oq, color=COLORS["oqpsk"],  label="O-QPSK (IEEE 802.15.4)", linestyle="-.")
    ax.semilogy(ebn0_db, ber_f,  color=COLORS["fsk"],    label="Non-coherent FSK", linestyle=":")

    # Target BER line for WSN
    ax.axhline(1e-3, color="gray", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.text(12.5, 1.4e-3, "BER = 10⁻³\n(WSN target)", fontsize=9, color="gray", ha="right")

    ax.set_xlabel("$E_b/N_0$ (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("Figure 1 — BER vs $E_b/N_0$: Modulation Comparison in AWGN Channel", fontweight="bold")
    ax.legend(loc="upper right")
    ax.set_xlim(-4, 14)
    ax.set_ylim(1e-6, 1)

    # Annotate Eb/N0 needed for BER=1e-3
    for label, ber_fn, color in [("BPSK", ber_bpsk, COLORS["bpsk"]),
                                   ("QPSK / O-QPSK", ber_qpsk, COLORS["oqpsk"])]:
        idx = np.argmin(np.abs(ber_fn(ebn0_lin) - 1e-3))
        ax.annotate(f"{label}\n{ebn0_db[idx]:.1f} dB",
                    xy=(ebn0_db[idx], 1e-3),
                    xytext=(ebn0_db[idx] - 2.5, 5e-5),
                    fontsize=8.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    fig.tight_layout()
    path = os.path.join(OUT, "fig1_ber_vs_ebno.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Received SNR vs Distance (Friis Path Loss)
# ═══════════════════════════════════════════════════════════════════════════════

def path_loss_db(d, freq_hz=2.4e9, n=2.0, d0=1.0):
    """Log-distance path loss model (dB). n=2 free space, n=3.5 indoor."""
    c = 3e8
    lam = c / freq_hz
    PL0 = 20 * np.log10(4 * np.pi * d0 / lam)      # free-space PL at reference d0
    return PL0 + 10 * n * np.log10(d / d0)

def plot_snr_vs_distance():
    # IEEE 802.15.4 / TI CC2420 parameters
    Pt_dBm   = 0         # Transmit power (dBm)
    NF_dB    = 9         # Receiver noise figure (dB) — CC2420 datasheet
    B_Hz     = 2e6       # Channel bandwidth (Hz)
    k_dBmHz  = -174      # Thermal noise density (dBm/Hz) at 290 K

    # Noise floor
    N_floor_dBm = k_dBmHz + 10 * np.log10(B_Hz) + NF_dB  # ≈ -101 dBm

    distances = np.logspace(np.log10(1), np.log10(300), 400)   # 1 m to 300 m

    snr_outdoor = Pt_dBm - path_loss_db(distances, n=2.0) - N_floor_dBm
    snr_indoor  = Pt_dBm - path_loss_db(distances, n=3.5) - N_floor_dBm

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.semilogx(distances, snr_outdoor, color=COLORS["outdoor"], label="Outdoor / Free Space (n = 2.0)")
    ax.semilogx(distances, snr_indoor,  color=COLORS["indoor"],  label="Indoor / Obstructed (n = 3.5)", linestyle="--")

    # Minimum SNR for BER < 10⁻³ with O-QPSK ≈ 7 dB Eb/N0; convert to SNR
    # SNR = Eb/N0 + 10*log10(Rb/B) = 7 + 10*log10(250e3/2e6) ≈ 7 - 9 = -2 dB
    min_snr = -2
    ax.axhline(min_snr, color="red", linewidth=1.4, linestyle=":", alpha=0.8)
    ax.text(200, min_snr + 1.5, "Min SNR for BER < 10⁻³\n(O-QPSK)", fontsize=9, color="red", ha="right")

    # Shade unusable region
    ax.fill_between(distances, min_snr, ax.get_ylim()[0] if ax.get_ylim()[0] < -30 else -30,
                    alpha=0.08, color="red")

    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Received SNR (dB)")
    ax.set_title("Figure 2 — Received SNR vs Distance\n(IEEE 802.15.4: $P_t$ = 0 dBm, 2.4 GHz)", fontweight="bold")
    ax.legend()
    ax.set_xlim(1, 300)

    # Annotate coverage limits
    for snr_arr, color, label in [(snr_outdoor, COLORS["outdoor"], "Outdoor"),
                                   (snr_indoor,  COLORS["indoor"],  "Indoor")]:
        idx = np.argmin(np.abs(snr_arr - min_snr))
        if 0 < idx < len(distances) - 1:
            ax.annotate(f"{label}\n{distances[idx]:.0f} m",
                        xy=(distances[idx], min_snr),
                        xytext=(distances[idx] * 1.5, min_snr + 6),
                        fontsize=9, color=color,
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    fig.tight_layout()
    path = os.path.join(OUT, "fig2_snr_vs_distance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Shannon Capacity vs SNR
# ═══════════════════════════════════════════════════════════════════════════════

def plot_shannon_capacity():
    snr_db  = np.linspace(-10, 30, 500)
    snr_lin = 10 ** (snr_db / 10)
    B_Hz    = 2e6        # 802.15.4 bandwidth

    capacity_bps = B_Hz * np.log2(1 + snr_lin)
    actual_rate  = 250e3   # 802.15.4 PHY data rate (bps)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(snr_db, capacity_bps / 1e3, color=COLORS["theory"], label="Shannon Capacity Limit  $C = B\\,\\log_2(1 + SNR)$")
    ax.axhline(actual_rate / 1e3, color=COLORS["actual"], linestyle="--", linewidth=2,
               label=f"IEEE 802.15.4 actual rate ({actual_rate/1e3:.0f} kbps)")

    # Find the SNR at which capacity = actual rate
    idx = np.argmin(np.abs(capacity_bps - actual_rate))
    min_snr_for_rate = snr_db[idx]
    ax.axvline(min_snr_for_rate, color="gray", linestyle=":", linewidth=1.4)
    ax.text(min_snr_for_rate + 0.5, capacity_bps.max() / 1e3 * 0.85,
            f"Min SNR = {min_snr_for_rate:.1f} dB\nfor 250 kbps",
            fontsize=9, color="gray")

    # Shade the "gap" — wasted spectral efficiency
    snr_high = snr_db[snr_db >= min_snr_for_rate]
    cap_high = capacity_bps[snr_db >= min_snr_for_rate] / 1e3
    ax.fill_between(snr_high, actual_rate / 1e3, cap_high, alpha=0.12,
                    color=COLORS["theory"], label="Unutilized capacity (system gap)")

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Channel Capacity / Data Rate (kbps)")
    ax.set_title("Figure 3 — Shannon Capacity vs SNR\n(B = 2 MHz, IEEE 802.15.4)", fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xlim(-10, 30)
    ax.set_ylim(0, capacity_bps.max() / 1e3 * 1.1)

    fig.tight_layout()
    path = os.path.join(OUT, "fig3_shannon_capacity.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Energy per Bit vs BER (Energy-Reliability Trade-off)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_energy_ber_tradeoff():
    """
    Energy per bit (J/bit) vs achieved BER for O-QPSK.
    TX power swept from -25 dBm to +5 dBm (typical WSN range).
    Link distance fixed at 30 m (typical indoor WSN hop).
    """
    Pt_dBm_range = np.linspace(-25, 5, 300)
    Pt_W         = 10 ** (Pt_dBm_range / 10) * 1e-3

    # Circuit power (leakage + baseband) — CC2420 datasheet ~18.8 mA @ 1.8 V ≈ 34 mW TX on
    P_circuit_W = 34e-3     # fixed circuit power during TX
    R_b         = 250e3     # bit rate (bps)

    # Energy per bit
    E_bit = (Pt_W + P_circuit_W) / R_b     # Joules per bit

    # SNR at receiver for 30 m indoor link
    d   = 30
    NF  = 9          # dB
    B   = 2e6        # Hz
    N_floor_dBm = -174 + 10 * np.log10(B) + NF   # ≈ -101 dBm
    N_floor_W   = 10 ** (N_floor_dBm / 10) * 1e-3

    # Path loss at 30 m indoor (n=3.5)
    PL_dB = path_loss_db(np.array([d]), n=3.5)[0]
    Pr_W  = Pt_W * 10 ** (-PL_dB / 10)

    SNR_lin = Pr_W / N_floor_W
    Eb_N0   = SNR_lin * (B / R_b)      # Eb/N0 linear
    ber     = ber_oqpsk(Eb_N0)
    ber     = np.clip(ber, 1e-9, 1)

    fig, ax1 = plt.subplots(figsize=(9, 6))

    color_e   = COLORS["oqpsk"]
    color_ber = COLORS["bpsk"]

    ax1.semilogy(Pt_dBm_range, ber, color=color_ber, label="BER (O-QPSK)")
    ax1.set_xlabel("Transmit Power $P_t$ (dBm)")
    ax1.set_ylabel("Bit Error Rate (BER)", color=color_ber)
    ax1.tick_params(axis='y', labelcolor=color_ber)
    ax1.set_ylim(1e-8, 1)

    ax2 = ax1.twinx()
    ax2.plot(Pt_dBm_range, E_bit * 1e6, color=color_e, linestyle="--", label="Energy/bit (µJ/bit)")
    ax2.set_ylabel("Energy per Bit (µJ/bit)", color=color_e)
    ax2.tick_params(axis='y', labelcolor=color_e)

    # Target BER line
    ax1.axhline(1e-3, color="gray", linestyle=":", linewidth=1.2, alpha=0.7)
    ax1.text(-23, 1.5e-3, "BER target 10⁻³", fontsize=9, color="gray")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Figure 4 — Energy–Reliability Trade-off (O-QPSK, d = 30 m indoor)\n"
              "Higher TX power → lower BER but higher energy consumption", fontweight="bold")

    fig.tight_layout()
    path = os.path.join(OUT, "fig4_energy_ber_tradeoff.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — BER vs Distance for all modulations
# ═══════════════════════════════════════════════════════════════════════════════

def plot_ber_vs_distance():
    distances = np.logspace(np.log10(1), np.log10(200), 400)

    Pt_dBm  = 0
    NF_dB   = 9
    B_Hz    = 2e6
    R_b     = 250e3

    N_floor_dBm = -174 + 10 * np.log10(B_Hz) + NF_dB
    N_floor_W   = 10 ** (N_floor_dBm / 10) * 1e-3

    def compute_ber(ber_fn, n_path):
        PL_dB = path_loss_db(distances, n=n_path)
        Pr_W  = 10 ** ((Pt_dBm - PL_dB) / 10) * 1e-3
        SNR   = Pr_W / N_floor_W
        EbN0  = SNR * (B_Hz / R_b)
        return np.clip(ber_fn(EbN0), 1e-10, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

    for ax, n, label in [(axes[0], 2.0, "Outdoor — Free Space (n = 2.0)"),
                          (axes[1], 3.5, "Indoor — Obstructed (n = 3.5)")]:
        ax.semilogy(distances, compute_ber(ber_bpsk,  n), color=COLORS["bpsk"],  label="BPSK")
        ax.semilogy(distances, compute_ber(ber_qpsk,  n), color=COLORS["qpsk"],  label="QPSK", linestyle="--")
        ax.semilogy(distances, compute_ber(ber_oqpsk, n), color=COLORS["oqpsk"], label="O-QPSK (802.15.4)", linestyle="-.")
        ax.semilogy(distances, compute_ber(ber_fsk_noncoherent, n),
                    color=COLORS["fsk"], label="Non-coherent FSK", linestyle=":")

        ax.axhline(1e-3, color="gray", linestyle=":", linewidth=1.2, alpha=0.8)
        ax.text(150, 1.6e-3, "BER = 10⁻³", fontsize=8.5, color="gray", ha="right")

        ax.set_xlabel("Distance (m)")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(1, 200)
        ax.set_ylim(1e-8, 1)

    axes[0].set_ylabel("Bit Error Rate (BER)")
    fig.suptitle("Figure 5 — BER vs Distance: Modulation Comparison\n"
                 "($P_t$ = 0 dBm, 2.4 GHz, B = 2 MHz)", fontweight="bold", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "fig5_ber_vs_distance.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# NUMERIC TABLES — print to console for use in report
# ═══════════════════════════════════════════════════════════════════════════════

def print_ber_table():
    print("\n-- BER Table: Modulation vs Eb/N0 ----------------------------------")
    print(f"{'Eb/N0 (dB)':>12} {'BPSK':>12} {'QPSK':>12} {'O-QPSK':>12} {'NC-FSK':>12}")
    print("-" * 62)
    for ebn0_db in [-10, -5, 0, 3, 5, 7, 10, 12]:
        ebn0 = 10 ** (ebn0_db / 10)
        row = [ebn0_db,
               ber_bpsk(ebn0),
               ber_qpsk(ebn0),
               ber_oqpsk(ebn0),
               ber_fsk_noncoherent(ebn0)]
        print(f"{row[0]:>12} {row[1]:>12.2e} {row[2]:>12.2e} {row[3]:>12.2e} {row[4]:>12.2e}")

def print_coverage_table():
    print("\n-- Coverage Range for BER < 1e-3 -----------------------------------")
    Pt_dBm = 0
    NF_dB  = 9
    B_Hz   = 2e6
    R_b    = 250e3
    N_floor_dBm = -174 + 10 * np.log10(B_Hz) + NF_dB
    N_floor_W   = 10 ** (N_floor_dBm / 10) * 1e-3
    distances = np.logspace(0, np.log10(500), 2000)

    for modname, ber_fn in [("BPSK", ber_bpsk), ("QPSK/O-QPSK", ber_qpsk), ("NC-FSK", ber_fsk_noncoherent)]:
        for env, n in [("Outdoor", 2.0), ("Indoor", 3.5)]:
            PL_dB = path_loss_db(distances, n=n)
            Pr_W  = 10 ** ((Pt_dBm - PL_dB) / 10) * 1e-3
            EbN0  = (Pr_W / N_floor_W) * (B_Hz / R_b)
            ber   = ber_fn(EbN0)
            # Last distance where BER < 1e-3
            valid = distances[ber < 1e-3]
            dmax  = valid[-1] if len(valid) > 0 else 0
            print(f"  {modname:>12} | {env:<8} | max range = {dmax:>6.1f} m")

def print_shannon_efficiency():
    print("\n-- Shannon Efficiency at typical WSN operating SNR -----------------")
    B = 2e6
    R_actual = 250e3
    for snr_db in [0, 5, 10, 15, 20]:
        snr = 10 ** (snr_db / 10)
        C   = B * np.log2(1 + snr)
        eff = R_actual / C * 100
        print(f"  SNR = {snr_db:>3} dB | C = {C/1e3:>7.1f} kbps | Efficiency = {eff:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("WSN Communication Theory Simulation")
    print("=" * 50)
    print("Generating figures ...")

    plot_ber_vs_ebno()
    plot_snr_vs_distance()
    plot_shannon_capacity()
    plot_energy_ber_tradeoff()
    plot_ber_vs_distance()

    print_ber_table()
    print_coverage_table()
    print_shannon_efficiency()

    print("\nAll done. Figures saved to:", OUT)
