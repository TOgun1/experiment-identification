"""Render the Phase 0 test image: an orange ball next to an A4 reference sheet."""

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "phase0_ball.png"


def main() -> None:
    img = np.full((480, 640, 3), 235, dtype=np.uint8)  # light grey background
    # A4 sheet (0.297 m long side) drawn upright on the left as a reference object.
    cv2.rectangle(img, (60, 90), (60 + 140, 90 + 198), (255, 255, 255), -1)
    cv2.rectangle(img, (60, 90), (60 + 140, 90 + 198), (40, 40, 40), 2)
    cv2.circle(img, (420, 200), 32, (0, 140, 255), -1, lineType=cv2.LINE_AA)  # BGR orange
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), img)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
