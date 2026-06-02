```bash
# model_sizes.pdf
uv run python -m analysis.model_sizes

# language_coverage.pdf
uv run python -m analysis.language_coverage

# collaboration_sectors.pdf
uv run python -m analysis.collaboration_sectors

# nlp_literature_by_stage_prop.pdf
uv run python -m analysis.nlp_literature

# infra_lingdiv_network.pdf AND infra_lingdiv_ict.pdf  (one run writes both; needs internet)
uv run python -m analysis.infra_lingdiv

# domain_method_network.pdf AND domain_method_network.png  (one run writes both)
uv run python -m analysis.deployment_domains

# literature_clusters_umap.pdf  (downloads all-MiniLM-L6-v2 on first run)
uv run python -m analysis.literature_clusters
```
