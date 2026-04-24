# website

Static project page. Plain HTML/CSS/JS, no build step.

## run locally

```
python -m http.server --directory website 8000
```

Open <http://localhost:8000>.

## edit pipeline copy

All paper text lives at the top of [`js/pipeline.js`](js/pipeline.js):

- `PIPELINE_STAGES` — per-stage `title` + `body`. Use `[1]`, `[2]` inline.
- `REFERENCES` — numbered 1-indexed, matches the markers.

Save, refresh.

## deploy (GitHub Pages)

Settings → Pages → Source: `main` branch, folder: `/website`.
