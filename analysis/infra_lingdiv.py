from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from analysis.utils import COLORS, PLOT_PARAMS, OUTPUT_DIR

CWD = Path(__file__).resolve().parent
ROOT = CWD.parent

plt.rcParams.update(PLOT_PARAMS)

INCOME_STYLE = {
    "Low-income countries": {"color": COLORS["cherry"], "marker": "o"},
    "Lower-middle-income countries": {"color": COLORS["purple"], "marker": "s"},
    "Upper-middle-income countries": {"color": COLORS["indigo"], "marker": "D"},
    "High-income countries": {"color": COLORS["green"], "marker": "^"},
}
INCOME_ORDER = [
    "Low-income countries",
    "Lower-middle-income countries",
    "Upper-middle-income countries",
    "High-income countries",
]


def main():
    income_df = pd.read_csv(
        "https://ourworldindata.org/grapher/world-bank-income-groups.csv?v=1&csvType=full&useColumnShortNames=true",
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"},
    )
    ict_adoption_df = pd.read_csv(
        "https://ourworldindata.org/grapher/ict-adoption-per-100-people.csv?v=1&csvType=full&useColumnShortNames=true",
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"},
    )
    network_df = pd.read_csv(
        "https://ourworldindata.org/grapher/population-covered-by-mobile-network-by-network-capability.csv?v=1&csvType=full&useColumnShortNames=true",
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"},
    )
    language_df = pd.read_csv(ROOT / "data" / "living_languages_per_country_2025.csv")

    income_latest = (
        income_df.sort_values("year")
        .groupby("entity", as_index=False)
        .last()[["entity", "classification"]]
        .rename(columns={"entity": "country", "classification": "income_group"})
    )
    # Filter to only countries that appear in World Bank income groups
    language_df = language_df[language_df["country"].isin(income_latest["country"])]
    network_2023 = (
        network_df[network_df["year"] == 2023][["entity", "_9_c_1__it_mob_4gntwk"]]
        .rename(
            columns={"entity": "country", "_9_c_1__it_mob_4gntwk": "network_access"}
        )
        .dropna(subset=["network_access"])
    )
    ict_2023 = (
        ict_adoption_df[ict_adoption_df["year"] == 2023][["entity", "it_net_user_zs"]]
        .rename(columns={"entity": "country", "it_net_user_zs": "internet_users"})
        .dropna(subset=["internet_users"])
    )

    # Network adoption
    df_net = network_2023.merge(language_df, on="country").merge(
        income_latest, on="country"
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    for group in INCOME_ORDER:
        style = INCOME_STYLE[group]
        subset = df_net[df_net["income_group"] == group]
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
    ax.set_yscale("log")
    ax.set_xlabel(r"\% population covered by 4G network")
    ax.set_ylabel("Number of living languages")
    ax.legend(frameon=False)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "infra_lingdiv_network.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved to outputs/infra_lingdiv_network.pdf")

    # ICT Adoption
    df_ict = ict_2023.merge(language_df, on="country").merge(
        income_latest, on="country"
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    for group in INCOME_ORDER:
        style = INCOME_STYLE[group]
        subset = df_ict[df_ict["income_group"] == group]
        ax.scatter(
            subset["internet_users"],
            subset["num_living_languages"],
            c=style["color"],
            marker=style["marker"],
            edgecolors=COLORS["slate_3"],
            linewidths=0.5,
            label=group.replace(" countries", ""),
            s=60,
            alpha=0.85,
        )
    # Annotate notable outliers
    annotate_countries = [
        "Papua New Guinea",
        "Nigeria",
        "Indonesia",
        "India",
        "China",
        "Cameroon",
        "Chad",
        "Democratic Republic of Congo",
        "Ethiopia",
        "United States",
    ]
    for _, row in df_ict[df_ict["country"].isin(annotate_countries)].iterrows():
        label = row["country"]
        if label == "Democratic Republic of Congo":
            label = "DR Congo"
        color = INCOME_STYLE[row["income_group"]]["color"]
        ax.annotate(
            label,
            (row["internet_users"], row["num_living_languages"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=12,
            color=color,
        )

    ax.set_yscale("log")
    ax.set_xlabel(r"\% Individuals using the Internet")
    ax.set_ylabel("Number of living languages")
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "infra_lingdiv_ict.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved to outputs/infra_lingdiv_ict.pdf")


if __name__ == "__main__":
    main()
