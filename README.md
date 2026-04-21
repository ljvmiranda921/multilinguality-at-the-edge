<img src="assets/cambridge_logo.png" height="80" />&nbsp;&nbsp;&nbsp;<img src="assets/ltl_logo2.svg" height="90" />

# Multilinguality at the Edge: Developing Language Models for the Global South

Where and how language models (LMs) are deployed determines who can benefit from them. 
However, there are several challenges that prevent effective deployment of LMs in non-English-speaking and hardware-constrained communities in the Global South.
We call this challenge the last mile: the intersection of multilinguality and edge deployment, where the goals are aligned but the technical requirements often compete.
Studying these two fields together is both a need, as linguistically diverse communities often face the most severe infrastructure constraints, 
and an opportunity, 
as edge and multilingual NLP research remain largely siloed.

To understand the state of the art and the challenges of combining the two areas, we survey 232 papers that tackle this problem across the language modelling pipeline, from
data collection to development and deployment. This repository contains all the experiments and analyses for this work and serves as an accompaniment for the survey paper, ["Multilinguality at the Edge: Developing Language Models for the Global South"]().

## Installation & Usage

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
git clone git@github.com:ljvmiranda921/multilingual-edge-nlp.git
cd multilingual-edge-nlp
uv sync
```

To generate figures, run `python -m analysis.<script_name>`.

To scrape papers from Semantic Scholar, run `python -m scripts.s2_scrape`.
You can optionally set `S2_API_KEY` in `.env` for higher rate limits, but unauthenticated requests work fine for moderate volumes.
Use `--query_names` to select specific queries (`multilingual`, `efficient`, `intersection`), or omit to run all three.

```bash
python -m scripts.s2_scrape --year 2020 --min_citations 5
```

To annotate papers using an LLM, run `python -m scripts.llm_annotate`.
We use Azure OpenAI by default&mdash;set `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY` in `.env`, then pass `--use_azure` with the deployment name as `--model`:

```bash
python -m scripts.llm_annotate --use_azure --model gpt-4.1-mini
```

You can also use the OpenAI API directly by setting `OPENAI_API_KEY` in `.env` and omitting the `--use_azure` flag.

## Data Sources

In this section, we list down the data sources to build some of the supporting figures.
You should be able to replicate the figures by running `python -m analysis.<script_name>`.

- [Share of population in range of mobile network](https://ourworldindata.org/grapher/population-covered-by-mobile-network-by-network-capability): [International Telecommunication Union](https://unstats.un.org/sdgs/dataportal), processed by [Our World in Data](https://ourworldindata.org/)
- [Living languages per country](https://ourworldindata.org/grapher/living-languages): [SIL International, Ethnologue (28th edition)](https://www.ethnologue.com/), processed by [Our World in Data](https://ourworldindata.org/)
- [ICT adoption per 100 people](https://ourworldindata.org/grapher/ict-adoption-per-100-people): [International Telecommunication Union](https://www.itu.int/) via [World Bank World Development Indicators](https://databank.worldbank.org/source/world-development-indicators), processed by [Our World in Data](https://ourworldindata.org/)
- [World Bank income groups](https://ourworldindata.org/grapher/world-bank-income-groups): [World Bank Country and Lending Groups](https://datahelpdesk.worldbank.org/knowledgebase/articles/906519-world-bank-country-and-lending-groups), processed by [Our World in Data](https://ourworldindata.org/)
- Research Papers were downloaded using the [Semantic Scholar API](https://www.semanticscholar.org/product/api).

## Acknowledgements

LJVM and AK acknowledge the support of the UKRI Frontier Grant EP/Y031350/1 ([EQUATE](https://gtr.ukri.org/projects?ref=EP%2FY031350%2F1)).
