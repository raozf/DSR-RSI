"""Deterministic OpenDM inference image preprocessing without training extras."""

import cv2
import numpy as np


class NoAugmentationPipeline:
    """Match upstream zero-padding then 448×448 bilinear resize."""

    def __call__(self, *, image: np.ndarray) -> dict[str, np.ndarray]:
        height, width = image.shape[:2]
        size = max(height, width)
        top, left = (size - height) // 2, (size - width) // 2
        padded = cv2.copyMakeBorder(
            image, top, size - height - top, left, size - width - left,
            cv2.BORDER_CONSTANT, value=0,
        )
        return {"image": cv2.resize(padded, (448, 448), interpolation=cv2.INTER_LINEAR)}
