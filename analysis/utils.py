from pathlib import Path

import torch


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


FONT_SIZES = {"small": 14, "medium": 18, "large": 24}

PLOT_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times"],
    "font.size": FONT_SIZES.get("medium"),
    "axes.titlesize": FONT_SIZES.get("large"),
    "axes.labelsize": FONT_SIZES.get("large"),
    "xtick.labelsize": FONT_SIZES.get("large"),
    "ytick.labelsize": FONT_SIZES.get("large"),
    "legend.fontsize": FONT_SIZES.get("medium"),
    "figure.titlesize": FONT_SIZES.get("medium"),
    "text.usetex": True,
}

# https://www.cam.ac.uk/brand-resources/guidelines/colour (with the help of claude)
COLORS = {
    # Core palette
    "cambridge_blue": "#8EE8D8",
    "light_blue": "#D1F9F1",
    "warm_blue": "#00BDB6",
    "dark_blue": "#133844",
    # Secondary palette - Crest
    "light_crest": "#FFE2C8",
    "warm_crest": "#FFC392",
    "crest": "#FD8153",
    "dark_crest": "#DD3025",
    # Secondary palette - Cherry
    "light_cherry": "#F2CAD8",
    "warm_cherry": "#E18AAC",
    "cherry": "#CD3572",
    "dark_cherry": "#911449",
    # Secondary palette - Purple
    "light_purple": "#F2ECF8",
    "warm_purple": "#D1B7EB",
    "purple": "#A368DF",
    "dark_purple": "#681FB1",
    # Secondary palette - Indigo
    "light_indigo": "#EBEDFB",
    "warm_indigo": "#B0B9F1",
    "indigo": "#5366E0",
    "dark_indigo": "#29347A",
    # Secondary palette - Green
    "light_green": "#DFF2EA",
    "warm_green": "#AFDFCB",
    "green": "#4DB78C",
    "dark_green": "#13553A",
    # Greyscale
    "white": "#FFFFFF",
    "slate_1": "#ECEEF1",
    "slate_2": "#B5BDC8",
    "slate_3": "#546072",
    "slate_4": "#232830",
    # Heritage & Restricted
    "heritage": "#85B09A",
    "judge_yellow": "#FFB81C",
    # Legacy aliases for backwards compatibility
    "blue": "#D1F9F1",
    "slate": "#B5BDC8",
}

OUTPUT_DIR = Path("plot_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----- Website styling for inline-SVG export -----

WEB_COLORS = {
    "ink":        "#35302E",
    "accent":     "#254EFF",
    "warm":       "#C96A2E",
    "warm_light": "#E39A5F",
    "cool":       "#7B93B8",
    "cool_light": "#B4C0D2",
    "muted":      "#6b6358",
    "rule":       "#e5ddcb",
    "paper":      "#F8ECDA",
    "white":      "#FFFFFF",
}

WEB_PLOT_PARAMS = {
    "font.family":       "Univers",
    "font.size":         12,
    "axes.titlesize":    14,
    "axes.labelsize":    13,
    "axes.titleweight":  "normal",
    "axes.labelweight":  "normal",
    "axes.edgecolor":    WEB_COLORS["ink"],
    "axes.labelcolor":   WEB_COLORS["ink"],
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
    "xtick.color":       WEB_COLORS["ink"],
    "ytick.color":       WEB_COLORS["ink"],
    "legend.fontsize":   11,
    "figure.titlesize":  13,
    "text.color":        WEB_COLORS["ink"],
    "text.usetex":       False,
    "svg.fonttype":      "none",
    "figure.facecolor":  "none",
    "axes.facecolor":    "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "grid.color":        WEB_COLORS["rule"],
}

WEB_TITLE_FONT = {"family": "Tomato Grotesk", "size": 13}

WEB_FIGURES_DIR = Path("docs/assets/figures")
WEB_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
