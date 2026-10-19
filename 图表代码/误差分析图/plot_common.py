from pathlib import Path
import matplotlib as mpl
from matplotlib import font_manager


def configure_cjk_rendering():
    preferred = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
        "Source Han Sans SC", "PingFang SC", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    selected = next((name for name in preferred if name in available), "DejaVu Sans")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [selected, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


CHART_EXPORT_DIR = Path(__file__).resolve().parent / "输出图片"
CHART_EXPORT_DIR.mkdir(exist_ok=True)
