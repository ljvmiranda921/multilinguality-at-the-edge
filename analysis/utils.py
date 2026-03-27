from pathlib import Path
import torch


def get_device() -> str:
    """Get the best available device (MPS for Mac, CUDA, or CPU)."""
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
