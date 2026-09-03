from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


ORIENTATION_INPUT = (126, 176)  # width, height (matches training downscale)
TEXT_STRIP_INPUT = (256, 48)  # width, height


class OrientationNet(nn.Module):
    """Lightweight up/down (180-degree) classifier for a rectified card."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(256, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def load_orientation_classifier(path: str | Path, device: torch.device) -> OrientationNet:
    model = OrientationNet().to(device)
    model.load_state_dict(torch.load(str(path), map_location=device))
    model.eval()
    return model


def predict_upright_probability(
    model: OrientationNet,
    image_bgr: np.ndarray,
    device: torch.device,
) -> float:
    """Return P(upright) for a rectified card image."""

    width, height = ORIENTATION_INPUT
    img = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_AREA)
    tensor = (
        torch.from_numpy(img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0).to(device)
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    return float(probabilities[0].item())


def _extract_text_strips(
    image_bgr: np.ndarray,
    max_strips: int = 10,
) -> list[np.ndarray]:
    """Extract candidate horizontal text lines by horizontal-gradient density."""

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    density = np.abs(grad_x).mean(axis=1)
    kernel = 41
    smoothed = np.convolve(density, np.ones(kernel) / kernel, mode="same")
    peaks: list[int] = []
    for index in range(1, len(smoothed) - 1):
        if (
            smoothed[index] >= smoothed[index - 1]
            and smoothed[index] > smoothed[index + 1]
            and smoothed[index] > float(np.median(smoothed)) * 1.35
        ):
            peaks.append(index)
    peaks.sort(key=lambda index: -float(smoothed[index]))
    strip_height = TEXT_STRIP_INPUT[1] * 4
    strips: list[np.ndarray] = []
    for peak in peaks[:max_strips]:
        top = max(0, peak - strip_height // 2)
        bottom = min(image_bgr.shape[0], top + strip_height)
        if bottom - top < strip_height * 0.7:
            continue
        strips.append(image_bgr[top:bottom])
    return strips


def predict_text_upright_probability(
    model: OrientationNet,
    image_bgr: np.ndarray,
    device: torch.device,
) -> tuple[float | None, int]:
    """Return (mean P(text upright), number of text strips) for a card image."""

    strips = _extract_text_strips(image_bgr)
    if not strips:
        return None, 0
    width, height = TEXT_STRIP_INPUT
    batch = np.stack(
        [
            cv2.resize(strip, (width, height), interpolation=cv2.INTER_AREA)
            .transpose(2, 0, 1)
            .astype(np.float32)
            / 255.0
            for strip in strips
        ]
    )
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(batch).to(device)), dim=1)[:, 0]
    return float(probabilities.mean().item()), len(strips)
