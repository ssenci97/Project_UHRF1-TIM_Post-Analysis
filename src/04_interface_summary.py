###################################### individual residue heatmaps
def plot_residues(records, top, out, n_models, burial_cutoff):
    chains = sorted({r["key"][0] for r in records})
    if not chains:
        chains = ["none"]
    for chain in chains:
        n = min(top, sum(r["key"][0] == chain for r in records)) if chain != "none" else 1
        fig, ax = plt.subplots(figsize=(5.0, max(2.8, 1.5 + n * 0.32)))
        fig.subplots_adjust(left=0.30, right=0.98, top=0.82, bottom=0.08)
        if chain == "none":
            ax.text(0.5, 0.5, "No residue interactions", ha="center", va="center", transform=ax.transAxes, fontsize=FONT)
            ax.set_axis_off()
        else:
            draw_residue_heatmap(ax, records, chain, top, burial_cutoff)
        fig.suptitle(f"Residue occupancy · Chain {chain} · {n_models} models", fontsize=FONT + 3, x=0.02, ha="left", y=0.98)
        finish_figure(fig, out / f"{out.parent.name}_A_residue_occupancy_chain_{chain}.svg")
    return chains


