"""Phase 0 acceptance check: one SAM 3.1 segmentation and one Muse Spark reply.

Run it twice. The first run calls the Meta Model API; the second must be served
entirely from data/cache (it reports `misses=0`).

    python scripts/phase0_smoke.py
    PHYSLAB_OFFLINE=1 python scripts/phase0_smoke.py   # proves the cache alone suffices
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from meta_sam_parser import decode_mask_to_raster

from physlab.config import REPO_ROOT
from physlab.meta_client import MetaClient

DEFAULT_IMAGE = REPO_ROOT / "data" / "synthetic" / "phase0_ball.png"


def mask_centroid(record) -> tuple[float, float, int]:
    """Centroid (x, y) in source-image pixels and foreground pixel count of a mask record.

    Masks are box-local rasters; `record.bounds` places them in the source image.
    """
    mask = record.mask
    raster = np.frombuffer(decode_mask_to_raster(mask), dtype=np.uint8).reshape(
        mask.height, mask.width
    )
    ys, xs = np.nonzero(raster)
    if len(xs) == 0:
        return float("nan"), float("nan"), 0
    b = record.bounds
    sx = (b.right - b.left) / mask.width
    sy = (b.bottom - b.top) / mask.height
    return b.left + (xs.mean() + 0.5) * sx, b.top + (ys.mean() + 0.5) * sy, len(xs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--concept", default="orange ball")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    client = MetaClient()

    print(f"SAM ({client.settings.sam_model}): segmenting {args.concept!r} in {args.image.name}")
    result = client.segment_image(args.image, args.concept)
    boxes = {r.object_id: r for r in result.records if r.kind == "box"}
    masks = [r for r in result.records if r.kind == "mask"]
    print(f"  outcome={result.outcome.status} masks={len(masks)}")
    for record in masks:
        cx, cy, area = mask_centroid(record)
        box = boxes.get(record.object_id)
        conf = box.confidence if box is not None else None
        print(f"  object {record.object_id}: centroid=({cx:.1f}, {cy:.1f}) px, area={area} px, confidence={conf}")

    print(f"Muse Spark ({client.settings.muse_model}):")
    reply = client.chat([{"role": "user", "content": "Reply with the single word: ready"}])
    print(f"  {reply.strip()!r}")

    print(f"Cache: hits={client.stats['hits']} misses={client.stats['misses']}")
    if not masks:
        print("FAIL: SAM returned no masks", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
