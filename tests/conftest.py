import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def manifest():
    p = FIXTURES / "manifest.json"
    if not p.is_file():
        pytest.skip("fixtures missing — run: python tests/fetch_fixtures.py")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def fixture_cache():
    from cesium_for_blender.core import provider

    return provider.DiskCache(str(FIXTURES / "cache"))


@pytest.fixture(scope="session")
def terrain_tiles(manifest, fixture_cache):
    """{(z,x,y): raw bytes} for every terrain fixture."""
    out = {}
    for z, x, y in manifest["terrain_keys"]:
        data = fixture_cache.get_terrain(z, x, y)
        assert data is not None, f"fixture terrain {z}/{x}/{y} missing"
        out[(z, x, y)] = data
    return out
