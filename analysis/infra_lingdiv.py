from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils import COLORS, PLOT_PARAMS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

plt.rcParams.update(PLOT_PARAMS)


def main():
    income_df = pd.read_csv(
        "https://ourworldindata.org/grapher/world-bank-income-groups.csv?v=1&csvType=full&useColumnShortNames=true",
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"},
    )
    network_df = pd.read_csv(
        "https://ourworldindata.org/grapher/population-covered-by-mobile-network-by-network-capability.csv?v=1&csvType=full&useColumnShortNames=true",
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"},
    )
    language_df = pd.read_csv(ROOT / "data" / "living_languages_per_country_2025.csv")

    network_2023 = (
        network_df[network_df["year"] == 2023][["entity", "_9_c_1__it_mob_4gntwk"]]
        .rename(
            columns={"entity": "country", "_9_c_1__it_mob_4gntwk": "network_access"}
        )
        .dropna(subset=["network_access"])
    )
    income_latest = (
        income_df.sort_values("year")
        .groupby("entity", as_index=False)
        .last()[["entity", "classification"]]
        .rename(columns={"entity": "country", "classification": "income_group"})
    )

    # Merge all three
    df = network_2023.merge(language_df, on="country").merge(
        income_latest, on="country"
    )

    income_style = {
        "Low-income countries": {"color": COLORS["cherry"], "marker": "o"},
        "Lower-middle-income countries": {"color": COLORS["purple"], "marker": "s"},
        "Upper-middle-income countries": {"color": COLORS["indigo"], "marker": "D"},
        "High-income countries": {"color": COLORS["green"], "marker": "^"},
    }
    income_order = [
        "Low-income countries",
        "Lower-middle-income countries",
        "Upper-middle-income countries",
        "High-income countries",
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    for group in income_order:
        style = income_style[group]
        subset = df[df["income_group"] == group]
        ax.scatter(
            subset["network_access"],
            subset["num_living_languages"],
            c=style["color"],
            marker=style["marker"],
            edgecolors=COLORS["slate_3"],
            linewidths=0.5,
            label=group.replace(" countries", ""),
            s=60,
            alpha=0.85,
        )

    ax.set_xlabel(r"\% population covered by 4G network")
    ax.set_ylabel("Number of living languages")
    ax.legend(frameon=False)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "infra_lingdiv.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved to outputs/infra_lingdiv.pdf")


if __name__ == "__main__":
    main()
