import subprocess
import sys
from pathlib import Path


PACKAGE_DIR = Path(__file__).parent


def launch(script_name):
    subprocess.run([sys.executable, str(PACKAGE_DIR / script_name)], check=True)


if __name__ == "__main__":
    launch("figure9_hovmoller.py")
    launch("figure10_drying_rate.py")
    print("全部绘图完成。")
