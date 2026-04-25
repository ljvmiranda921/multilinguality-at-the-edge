# Website

Static project page. Plain HTML/CSS/JS, no build step.

## Run locally

```
python -m http.server --directory website 8000
```

Open <http://localhost:8000>.

## Edit pipeline copy

All paper text lives at the top of [`js/pipeline.js`](js/pipeline.js):

- `PIPELINE_STAGES` — per-stage `title` + `body`. Use `[1]`, `[2]` inline.
- `REFERENCES` — numbered 1-indexed, matches the markers.

Save, refresh.

## Deploy (GitHub Pages)

Settings → Pages → Source: `main` branch, folder: `/website`.

## Disclosure of AI use

A large part of the website's codebase was created using Claude and Codex, especially the interactive versions of the paper's charts (although the static versions are mine).
I triple-checked the information when generating the interactive figures, but please report back if you see any inconsistencies in the paper.