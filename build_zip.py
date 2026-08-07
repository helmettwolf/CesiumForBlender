"""Package the addon into dist/cesium_for_blender.zip for Blender's
Preferences > Add-ons > Install."""

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "cesium_for_blender"
DIST = ROOT / "dist"


def main():
    DIST.mkdir(exist_ok=True)
    out = DIST / "cesium_for_blender.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for py in sorted(PKG.rglob("*.py")):
            if "__pycache__" in py.parts:
                continue
            zf.write(py, py.relative_to(ROOT))
    names = zipfile.ZipFile(out).namelist()
    print(f"{out} ({out.stat().st_size:,} bytes, {len(names)} files)")
    for n in names:
        print(" ", n)


if __name__ == "__main__":
    main()
