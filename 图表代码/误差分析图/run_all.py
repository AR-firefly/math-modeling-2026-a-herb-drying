from pathlib import Path
import runpy

PACKAGE_DIR = Path(__file__).resolve().parent
for script_name in ["图1_整体误差指标.py", "图2_数值误差来源.py"]:
    runpy.run_path(str(PACKAGE_DIR / script_name), run_name="__main__")
print("两张图片均已生成，见“输出图片”文件夹。")
