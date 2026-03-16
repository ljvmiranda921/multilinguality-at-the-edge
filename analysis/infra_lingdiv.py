import pandas as pd
import requests


def main():

    # Process network capability
    income_df = pd.read_csv(
        "https://ourworldindata.org/grapher/world-bank-income-groups.csv?v=1&csvType=full&useColumnShortNames=true",
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"},
    )
    network_df = pd.read_csv(
        "https://ourworldindata.org/grapher/population-covered-by-mobile-network-by-network-capability.csv?v=1&csvType=full&useColumnShortNames=true",
        storage_options={"User-Agent": "Our World In Data data fetch/1.0"},
    )
    breakpoint()


if __name__ == "__main__":
    main()
