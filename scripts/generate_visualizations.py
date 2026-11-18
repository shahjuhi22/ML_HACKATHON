import os
import sys
sys.path.insert(0, os.path.abspath("."))
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

from pathsnap.data.loader import load_trip, list_trips
from pathsnap.data.candidates import generate_candidates, CORE_FEATS
from pathsnap.data.alignment import proxy_reference_labels
from pathsnap.hmm.decoder import HMMViterbiDecoder
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_ROOT = os.environ.get("PATHSNAP_DATA_ROOT", os.path.join(ROOT, "data"))


def generate_all_visualizations():
    base_data = os.path.join(DATA_ROOT)
    res_dir = os.path.join(ROOT, "results")
    fig_dir = os.path.join(res_dir, "figures")
    map_dir = os.path.join(res_dir, "maps")
    tab_dir = os.path.join(res_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(map_dir, exist_ok=True)

    print("Generating Folium Maps...")
    # 1. Interactive Folium Map for Training Trip 00000001
    track, nodes, arcs, route = load_trip(os.path.join(base_data, "train"), "00000001", with_route=True)
    cands = generate_candidates(track, nodes, arcs)
    
    # Run HMM on trip 1
    decoder = HMMViterbiDecoder(sigma=10.0)
    pred_arcs = decoder.decode(cands, arcs, track[:, 2], len(track))

    center_lat = float(track[:, 1].mean())
    center_lon = float(track[:, 0].mean())
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: GPS Points
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Every 5th point)").add_to(m)
    for i in range(0, len(track), 5):
        folium.CircleMarker(
            location=[float(track[i, 1]), float(track[i, 0])],
            radius=3,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.7,
            popup=f"Obs {i}: Arc {pred_arcs[i]}"
        ).add_to(gps_group)

    # Layer 2: Reference Route (Green)
    ref_group = folium.FeatureGroup(name="Ground-Truth Reference Route").add_to(m)
    unique_route_arcs = np.unique(route)
    for a in unique_route_arcs:
        u, v = arcs[a, 0], arcs[a, 1]
        folium.PolyLine(
            locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
            color="#10b981",
            weight=4,
            opacity=0.8
        ).add_to(ref_group)

    # Layer 3: Predicted Route (Red Dashed)
    pred_group = folium.FeatureGroup(name="HMM/Viterbi Predicted Path").add_to(m)
    valid_preds = [a for a in pred_arcs if a >= 0]
    for a in set(valid_preds):
        u, v = arcs[a, 0], arcs[a, 1]
        folium.PolyLine(
            locations=[[nodes[u, 1], nodes[u, 0]], [nodes[v, 1], nodes[v, 0]]],
            color="#ef4444",
            weight=3,
            dash_array="5, 5",
            opacity=0.9
        ).add_to(pred_group)

    folium.LayerControl(collapsed=False).add_to(m)
    map1_path = os.path.join(map_dir, "map_00000001_interactive.html")
    m.save(map1_path)
    print(f"Saved Folium map to {map1_path}")

    # Also save to juhi's results folder
    m.save(os.path.join(ROOT, "results", "maps", "map_00000001_interactive.html"))

    # 2. Comparison Plots
    print("Generating Comparison Figures for Technical Report...")
    comp_csv = os.path.join(tab_dir, "model_comparison.csv")
    if os.path.isfile(comp_csv):
        df = pd.read_csv(comp_csv)

        # Plot 1: Model Comparison (Top-1 Accuracy & Macro F1)
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(df))
        width = 0.35
        ax.bar(x - width/2, df["test_top1_accuracy"] * 100, width, label="Test Top-1 Accuracy (%)", color="#2563eb")
        ax.bar(x + width/2, df["test_macro_f1"] * 100, width, label="Test Macro F1 (%)", color="#10b981")
        ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
        ax.set_title("Machine Learning Model Comparison on Student-Testing Partition", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", " ").title() for m in df["model"]], fontsize=10, rotation=15)
        ax.set_ylim(60, 90)
        ax.legend(frameon=True)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        for i in range(len(df)):
            ax.text(i - width/2, df["test_top1_accuracy"].iloc[i] * 100 + 0.8, f"{df['test_top1_accuracy'].iloc[i]*100:.1f}%", ha="center", fontsize=9)
            ax.text(i + width/2, df["test_macro_f1"].iloc[i] * 100 + 0.8, f"{df['test_macro_f1'].iloc[i]*100:.1f}%", ha="center", fontsize=9)
        plt.tight_layout()
        fig_path1 = os.path.join(fig_dir, "model_comparison_bar.png")
        plt.savefig(fig_path1, dpi=200)
        plt.close()
        print(f"Saved {fig_path1}")

        # Plot 2: Route Continuity & Precision/Recall
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df["model"], df["precision"] * 100, marker="o", linewidth=2, label="Precision (%)", color="#0284c7")
        ax.plot(df["model"], df["recall"] * 100, marker="s", linewidth=2, label="Recall (%)", color="#16a34a")
        ax.plot(df["model"], df["route_continuity"] * 100, marker="^", linewidth=2, linestyle="--", label="Route Continuity (%)", color="#dc2626")
        ax.set_ylabel("Metric Value (%)", fontsize=11, fontweight="bold")
        ax.set_title("Precision, Recall, and Route Continuity across Models", fontsize=12, fontweight="bold")
        ax.set_xticklabels([m.replace("_", " ").title() for m in df["model"]], rotation=15)
        ax.set_ylim(70, 95)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        fig_path2 = os.path.join(fig_dir, "precision_recall_continuity.png")
        plt.savefig(fig_path2, dpi=200)
        plt.close()
        print(f"Saved {fig_path2}")

    # Plot 3: HMM vs Baseline Bar Chart
    hmm_csv = os.path.join(tab_dir, "hmm_vs_baseline.csv")
    if os.path.isfile(hmm_csv):
        hdf = pd.read_csv(hmm_csv)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        test_row = hdf[hdf["partition"] == "student_test"].iloc[0]
        methods = ["Nearest Baseline", "Classifier Only", "Pure HMM", "Hybrid (Clf+HMM)"]
        accs = [test_row["baseline_accuracy"]*100, test_row["classifier_only_accuracy"]*100, test_row["pure_hmm_accuracy"]*100, test_row["hybrid_hmm_accuracy"]*100]
        conts = [test_row["baseline_continuity"]*100, test_row["classifier_only_continuity"]*100, test_row["pure_hmm_continuity"]*100, test_row["hybrid_hmm_continuity"]*100]

        ax1.bar(methods, accs, color=["#94a3b8", "#38bdf8", "#4ade80", "#22c55e"])
        ax1.set_title("Pointwise Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Accuracy (%)", fontsize=10)
        ax1.set_ylim(40, 95)
        ax1.set_xticklabels(methods, rotation=20, ha="right")
        ax1.grid(axis="y", linestyle="--", alpha=0.5)
        for i, v in enumerate(accs):
            ax1.text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

        ax2.bar(methods, conts, color=["#94a3b8", "#60a5fa", "#f59e0b", "#10b981"])
        ax2.set_title("Route Continuity (%)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Continuity (%)", fontsize=10)
        ax2.set_ylim(60, 105)
        ax2.set_xticklabels(methods, rotation=20, ha="right")
        ax2.grid(axis="y", linestyle="--", alpha=0.5)
        for i, v in enumerate(conts):
            ax2.text(i, v + 1.0, f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")

        plt.suptitle("Impact of Sequential HMM & Viterbi Decoding (Student-Test Partition)", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        fig_path3 = os.path.join(fig_dir, "hmm_vs_baseline_comparison.png")
        plt.savefig(fig_path3, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved {fig_path3}")

    # Plot 4: Optuna Convergence History
    opt_csv = os.path.join(tab_dir, "optuna_trials.csv")
    if os.path.isfile(opt_csv):
        odf = pd.read_csv(opt_csv)
        if "value" in odf.columns:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            values = odf["value"].values * 100
            best_cum = np.maximum.accumulate(values)
            ax.plot(odf["number"], values, "o", color="#3b82f6", alpha=0.6, label="Trial CV Score")
            ax.plot(odf["number"], best_cum, "-", color="#dc2626", linewidth=2.5, label="Best Cumulative Score")
            ax.set_xlabel("Trial Number", fontsize=10, fontweight="bold")
            ax.set_ylabel("3-Fold CV Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
            ax.set_title("Optuna Bayesian Optimization Convergence History", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(frameon=True)
            plt.tight_layout()
            fig_path4 = os.path.join(fig_dir, "optuna_convergence.png")
            plt.savefig(fig_path4, dpi=200)
            plt.close()
            print(f"Saved {fig_path4}")

    # Plot 5: Geographical Trajectory Matched Map Export (PNG)
    fig, ax = plt.subplots(figsize=(9, 8))
    # Plot road network nodes
    ax.scatter(nodes[::3, 0], nodes[::3, 1], s=0.3, c="#cbd5e1", label="Road Network Nodes", rasterized=True)
    # Plot route segments
    for a in unique_route_arcs[:200]:
        u, v = arcs[a, 0], arcs[a, 1]
        ax.plot([nodes[u, 0], nodes[v, 0]], [nodes[u, 1], nodes[v, 1]], color="#10b981", linewidth=2.5, alpha=0.8)
    # Plot GPS track
    ax.plot(track[:, 0], track[:, 1], "o-", color="#2563eb", markersize=2.5, linewidth=0.8, alpha=0.6, label="Raw GPS Track")
    ax.set_title("Trip 00000001: GPS Observations vs Ground Truth Route", fontsize=13, fontweight="bold")
    ax.set_xlabel("Longitude (deg)", fontsize=10)
    ax.set_ylabel("Latitude (deg)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig_path5 = os.path.join(fig_dir, "trip_trajectory_map.png")
    plt.savefig(fig_path5, dpi=200)
    plt.close()
    print(f"Saved {fig_path5}")

if __name__ == "__main__":
    generate_all_visualizations()
