from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EDGES = ("left", "right", "top", "bottom")
EDGE_TO_KEY = {
    "left": "x_left",
    "right": "x_right",
    "top": "y_top",
    "bottom": "y_bottom",
}
CANONICAL_SIGN = {
    "left": 1.0,
    "right": -1.0,
    "top": 1.0,
    "bottom": -1.0,
}


@dataclass(frozen=True)
class EdgePrediction:
    edge: str
    coarse: float
    refined: float
    offset: float
    confidence: float
    entropy: float
    peak_mass: float
    canonical_position: float
    tta_disagreement: float = 0.0


class ConvBlock(nn.Module):
    def __init__(self, c1: int, c2: int, *, down_height: bool = False) -> None:
        super().__init__()
        stride = (2, 1) if down_height else (1, 1)
        groups = 8 if c2 % 8 == 0 else 4
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.GroupNorm(groups, c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EdgeRefiner(nn.Module):
    """Small high-resolution network that preserves the localization axis."""

    def __init__(self, input_channels: int = 4) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.tta_height_flip = False
        self.features = nn.Sequential(
            ConvBlock(input_channels, 24),
            ConvBlock(24, 32, down_height=True),
            ConvBlock(32, 40, down_height=True),
            ConvBlock(40, 56, down_height=True),
            ConvBlock(56, 64, down_height=True),
            nn.Conv2d(64, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.score = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score_map = self.score(self.features(x))
        return score_map.mean(dim=2).squeeze(1)


class ResidualConvBlock(nn.Module):
    def __init__(self, c1: int, c2: int, *, down_height: bool = False) -> None:
        super().__init__()
        stride = (2, 1) if down_height else (1, 1)
        groups = 8 if c2 % 8 == 0 else 4
        self.body = nn.Sequential(
            nn.Conv2d(c1, c2, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, c2),
            nn.SiLU(inplace=True),
            nn.Conv2d(c2, c2, 3, padding=1, bias=False),
            nn.GroupNorm(groups, c2),
        )
        if c1 == c2 and not down_height:
            self.skip: nn.Module = nn.Identity()
        else:
            self.skip = nn.Sequential(
                nn.Conv2d(c1, c2, 1, stride=stride, bias=False),
                nn.GroupNorm(groups, c2),
            )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.skip(x))


class DilatedAxisBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EdgeRefinerV5(nn.Module):
    """Boundary refiner with robust long-edge pooling and localization-axis context."""

    def __init__(self, input_channels: int = 7) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.tta_height_flip = True
        self.features = nn.Sequential(
            ResidualConvBlock(input_channels, 32),
            ResidualConvBlock(32, 40, down_height=True),
            ResidualConvBlock(40, 48, down_height=True),
            ResidualConvBlock(48, 64, down_height=True),
            ResidualConvBlock(64, 64, down_height=True),
        )
        self.pool_fusion = nn.Sequential(
            nn.Conv1d(128, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
        )
        self.axis_context = nn.Sequential(
            DilatedAxisBlock(64, 1),
            DilatedAxisBlock(64, 2),
            DilatedAxisBlock(64, 4),
            nn.Conv1d(64, 32, 1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
        )
        self.score = nn.Conv1d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = torch.cat((features.mean(dim=2), features.amax(dim=2)), dim=1)
        fused = self.pool_fusion(pooled)
        return self.score(self.axis_context(fused)).squeeze(1)


def _copy_box(box: dict[str, Any]) -> dict[str, float]:
    return {key: float(box[key]) for key in ("x_left", "x_right", "y_top", "y_bottom")}


def canonical_target_index(
    edge: str,
    target: float,
    coarse: float,
    *,
    band_half: float,
    patch_width: int,
) -> float:
    canonical_offset = CANONICAL_SIGN[edge] * (float(target) - float(coarse))
    canonical_x = float(band_half) + canonical_offset
    return canonical_x / (2.0 * float(band_half)) * float(patch_width - 1)


def index_to_edge_coordinate(
    edge: str,
    position: float,
    coarse: float,
    *,
    band_half: float,
    patch_width: int,
) -> float:
    canonical_x = float(position) / float(patch_width - 1) * (2.0 * float(band_half))
    canonical_offset = canonical_x - float(band_half)
    return float(coarse) + CANONICAL_SIGN[edge] * canonical_offset


def make_edge_patch(
    image: np.ndarray,
    edge: str,
    coarse: float,
    box: dict[str, Any],
    *,
    band_half: int = 32,
    patch_width: int = 96,
    patch_height: int = 256,
    long_pad_ratio: float = 0.008,
) -> np.ndarray:
    """Extract an edge strip and orient it as outside-left, inside-right."""

    if edge not in EDGES:
        raise ValueError(f"Unsupported edge: {edge}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a BGR image with three channels")

    height, width = image.shape[:2]
    current = _copy_box(box)
    border = band_half + 4
    padded = cv2.copyMakeBorder(image, border, border, border, border, cv2.BORDER_REFLECT_101)
    crop_span = band_half * 2 + 1

    if edge in ("left", "right"):
        long_pad = height * long_pad_ratio
        start = max(0.0, current["y_top"] - long_pad)
        end = min(float(height - 1), current["y_bottom"] + long_pad)
        long_size = max(32, int(round(end - start + 1.0)))
        center = (float(coarse) + border, (start + end) * 0.5 + border)
        patch = cv2.getRectSubPix(padded, (crop_span, long_size), center)
    else:
        long_pad = width * long_pad_ratio
        start = max(0.0, current["x_left"] - long_pad)
        end = min(float(width - 1), current["x_right"] + long_pad)
        long_size = max(32, int(round(end - start + 1.0)))
        center = ((start + end) * 0.5 + border, float(coarse) + border)
        horizontal = cv2.getRectSubPix(padded, (long_size, crop_span), center)
        patch = np.transpose(horizontal, (1, 0, 2))

    if edge in ("right", "bottom"):
        patch = patch[:, ::-1]
    patch = cv2.resize(patch, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(patch)


def augment_patch(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    image = patch.astype(np.float32)
    gain = float(rng.uniform(0.72, 1.30))
    bias = float(rng.uniform(-24.0, 24.0))
    image = image * gain + bias

    if rng.random() < 0.28:
        sigma = float(rng.uniform(0.35, 1.25))
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    if rng.random() < 0.32:
        image += rng.normal(0.0, rng.uniform(1.0, 7.0), image.shape).astype(np.float32)
    if rng.random() < 0.35:
        # Broad streaks approximate holographic glare while keeping the edge label unchanged.
        h, w = image.shape[:2]
        x = np.arange(w, dtype=np.float32)
        center = float(rng.uniform(0, w - 1))
        spread = float(rng.uniform(max(2.0, w * 0.04), max(4.0, w * 0.22)))
        stripe = np.exp(-0.5 * ((x - center) / spread) ** 2)
        strength = float(rng.uniform(-35.0, 65.0))
        image += stripe[None, :, None] * strength
    if rng.random() < 0.18:
        h, w = image.shape[:2]
        cut_h = int(rng.integers(max(2, h // 24), max(3, h // 8)))
        y0 = int(rng.integers(0, max(1, h - cut_h)))
        image[y0 : y0 + cut_h] = image.mean(axis=(0, 1), keepdims=True)
    return np.clip(image, 0, 255).astype(np.uint8)


def augment_bottom_logo_distractor(
    patch: np.ndarray,
    target_position: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add broken high-contrast lines outside a true bottom print boundary.

    Bottom copyright/trademark glyphs are parallel to the desired inner-frame
    edge after canonical strip rotation.  Their strokes are sharp but less
    continuous than the printed frame.  This augmentation makes that
    distinction an explicit hard negative while leaving the target unchanged.
    """

    if rng.random() >= 0.82:
        return patch
    image = patch.copy()
    height, width = image.shape[:2]
    target = float(np.clip(target_position, 5.0, width - 5.0))
    max_distance = min(26.0, target - 2.0)
    if max_distance < 5.0:
        return image
    center = int(round(target - float(rng.uniform(4.5, max_distance))))
    base_luma = float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
    if rng.random() < 0.68:
        level = int(rng.uniform(8, max(28.0, base_luma * 0.48)))
    else:
        level = int(rng.uniform(min(210.0, base_luma + 65.0), 255.0))
    color = (level, level, level)
    overlay = image.copy()

    # Several broken near-parallel strokes emulate a copyright line or logo,
    # but deliberately lack the full long-side continuity of the true frame.
    stroke_count = int(rng.integers(1, 4))
    for stroke in range(stroke_count):
        x = int(np.clip(center + stroke * int(rng.integers(1, 4)), 1, width - 2))
        segments = int(rng.integers(3, 8))
        for _ in range(segments):
            y0 = int(rng.integers(0, max(1, height - 12)))
            segment_height = int(rng.integers(max(4, height // 35), max(8, height // 8)))
            y1 = min(height - 1, y0 + segment_height)
            cv2.line(
                overlay,
                (x, y0),
                (x + int(rng.integers(-1, 2)), y1),
                color,
                int(rng.integers(1, 4)),
                cv2.LINE_AA,
            )
    for _ in range(int(rng.integers(5, 15))):
        x0 = int(np.clip(center + int(rng.integers(-4, 7)), 0, width - 2))
        y0 = int(rng.integers(0, max(1, height - 8)))
        x1 = int(np.clip(x0 + int(rng.integers(1, 6)), x0 + 1, width - 1))
        y1 = int(np.clip(y0 + int(rng.integers(2, 10)), y0 + 1, height - 1))
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)

    alpha = float(rng.uniform(0.45, 0.90))
    image = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)
    if rng.random() < 0.28:
        image = cv2.GaussianBlur(image, (0, 0), float(rng.uniform(0.25, 0.75)))
    return image


def augment_patch_v5(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Photometric-only corruption mix; the annotated edge coordinate stays fixed."""

    image = patch.astype(np.float32)
    gain = float(rng.uniform(0.58, 1.48))
    gamma = float(rng.uniform(0.68, 1.48))
    bias = float(rng.uniform(-38.0, 38.0))
    image = np.clip(image / 255.0, 0.0, 1.0) ** gamma
    image = image * (255.0 * gain) + bias

    if rng.random() < 0.46:
        channel_gain = rng.uniform(0.78, 1.24, (1, 1, 3)).astype(np.float32)
        image *= channel_gain
    if rng.random() < 0.42:
        sigma = float(rng.uniform(0.25, 1.8))
        image = cv2.GaussianBlur(image, (0, 0), sigma)
    if rng.random() < 0.20:
        kernel_size = int(rng.choice((3, 5, 7)))
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
        else:
            kernel[:, kernel_size // 2] = 1.0 / kernel_size
        image = cv2.filter2D(image, -1, kernel)
    if rng.random() < 0.48:
        image += rng.normal(0.0, rng.uniform(1.0, 11.0), image.shape).astype(np.float32)
    if rng.random() < 0.20:
        amount = float(rng.uniform(0.0005, 0.004))
        mask = rng.random(image.shape[:2])
        image[mask < amount * 0.5] = 0
        image[(mask >= amount * 0.5) & (mask < amount)] = 255
    if rng.random() < 0.58:
        height, width = image.shape[:2]
        x = np.arange(width, dtype=np.float32)
        for _ in range(int(rng.integers(1, 4))):
            center = float(rng.uniform(-0.1 * width, 1.1 * width))
            spread = float(rng.uniform(max(1.5, width * 0.025), max(4.0, width * 0.25)))
            stripe = np.exp(-0.5 * ((x - center) / spread) ** 2)
            color = rng.uniform(0.65, 1.35, (1, 1, 3)).astype(np.float32)
            strength = float(rng.uniform(-55.0, 110.0))
            image += stripe[None, :, None] * color * strength
    if rng.random() < 0.34:
        height, width = image.shape[:2]
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        direction = float(rng.uniform(0.0, 2.0 * np.pi))
        projection = xx * np.cos(direction) + yy * np.sin(direction)
        projection = (projection - projection.min()) / max(1e-6, float(np.ptp(projection)))
        strength = float(rng.uniform(-60.0, 45.0))
        image += projection[..., None] * strength
    if rng.random() < 0.30:
        height, width = image.shape[:2]
        cut_height = int(rng.integers(max(2, height // 32), max(3, height // 7)))
        y0 = int(rng.integers(0, max(1, height - cut_height)))
        fill = image.mean(axis=(0, 1), keepdims=True)
        image[y0 : y0 + cut_height] = fill

    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    if rng.random() < 0.34:
        quality = int(rng.integers(32, 91))
        ok, encoded = cv2.imencode(".jpg", image_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is not None:
                image_u8 = decoded
    return image_u8


def patch_to_tensor(patch: np.ndarray, *, input_channels: int = 4) -> torch.Tensor:
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    signed_gradient = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient = np.abs(signed_gradient)
    scale = float(np.percentile(gradient, 97.0))
    if scale > 1e-6:
        gradient = np.clip(gradient / scale, 0.0, 1.0)
        signed_gradient = np.clip(signed_gradient / scale, -1.0, 1.0)
    if input_channels == 4:
        features = np.concatenate((rgb * 2.0 - 1.0, gradient[..., None]), axis=2)
    elif input_channels == 7:
        local_mean = cv2.GaussianBlur(gray, (0, 0), 2.2)
        local_contrast = np.clip((gray - local_mean) * 4.0, -1.0, 1.0)
        features = np.concatenate(
            (
                rgb * 2.0 - 1.0,
                (gray * 2.0 - 1.0)[..., None],
                signed_gradient[..., None],
                gradient[..., None],
                local_contrast[..., None],
            ),
            axis=2,
        )
    else:
        raise ValueError(f"Unsupported edge-refiner input channels: {input_channels}")
    return torch.from_numpy(np.transpose(features, (2, 0, 1))).float()


def gaussian_target(
    positions: torch.Tensor,
    width: int,
    *,
    sigma: float = 1.6,
) -> torch.Tensor:
    coordinates = torch.arange(width, device=positions.device, dtype=positions.dtype)[None]
    targets = torch.exp(-0.5 * ((coordinates - positions[:, None]) / sigma) ** 2)
    return targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-9)


def localization_loss(
    logits: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    sigma: float = 1.6,
    regression_weight: float = 0.12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target_distribution = gaussian_target(target_positions, logits.shape[1], sigma=sigma)
    log_probabilities = F.log_softmax(logits, dim=1)
    distribution_loss = -(target_distribution * log_probabilities).sum(dim=1).mean()
    probabilities = log_probabilities.exp()
    coordinates = torch.arange(logits.shape[1], device=logits.device, dtype=logits.dtype)
    expected = (probabilities * coordinates[None]).sum(dim=1)
    regression_loss = F.smooth_l1_loss(expected, target_positions, beta=1.0)
    loss = distribution_loss + regression_weight * regression_loss
    return loss, {
        "distribution": distribution_loss.detach(),
        "regression": regression_loss.detach(),
        "expected": expected.detach(),
    }


@torch.inference_mode()
def predict_edge(
    model: EdgeRefiner,
    image: np.ndarray,
    edge: str,
    coarse: float,
    box: dict[str, Any],
    *,
    device: torch.device,
    band_half: int = 32,
    patch_width: int = 96,
    patch_height: int = 256,
) -> EdgePrediction:
    patch = make_edge_patch(
        image,
        edge,
        coarse,
        box,
        band_half=band_half,
        patch_width=patch_width,
        patch_height=patch_height,
    )
    input_channels = int(getattr(model, "input_channels", 4))
    tensor = patch_to_tensor(patch, input_channels=input_channels)
    if bool(getattr(model, "tta_height_flip", False)):
        batch = torch.stack((tensor, torch.flip(tensor, dims=(1,))), dim=0).to(device, non_blocking=True)
        batch_logits = model(batch)
        logits = batch_logits.mean(dim=0)
        batch_probabilities = batch_logits.softmax(dim=1)
        coordinates = torch.arange(patch_width, device=device, dtype=batch_probabilities.dtype)
        positions = (batch_probabilities * coordinates[None]).sum(dim=1)
        tta_disagreement = float((positions.max() - positions.min()).abs().item())
    else:
        logits = model(tensor[None].to(device, non_blocking=True))[0]
        tta_disagreement = 0.0
    probabilities = logits.softmax(dim=0)
    coordinates = torch.arange(patch_width, device=device, dtype=probabilities.dtype)
    expected_position = float((probabilities * coordinates).sum().item())
    peak_index = int(probabilities.argmax().item())
    local_position = float(peak_index)
    if 0 < peak_index < patch_width - 1:
        left_logit = float(logits[peak_index - 1].item())
        center_logit = float(logits[peak_index].item())
        right_logit = float(logits[peak_index + 1].item())
        denominator = left_logit - 2.0 * center_logit + right_logit
        if denominator < -1e-6:
            delta = 0.5 * (left_logit - right_logit) / denominator
            local_position += float(np.clip(delta, -0.5, 0.5))
    entropy = float((-(probabilities * probabilities.clamp_min(1e-9).log()).sum() / np.log(patch_width)).item())
    local_weight = float(np.clip((0.75 - entropy) / 0.35, 0.0, 0.72))
    position = (1.0 - local_weight) * expected_position + local_weight * local_position
    center_index = int(round(position))
    low, high = max(0, center_index - 2), min(patch_width, center_index + 3)
    peak_mass = float(probabilities[low:high].sum().item())
    disagreement_penalty = min(0.35, tta_disagreement / max(1.0, patch_width * 0.12))
    confidence = float(
        np.clip(0.55 * (1.0 - entropy) + 0.45 * peak_mass - disagreement_penalty, 0.0, 1.0)
    )
    refined = index_to_edge_coordinate(
        edge,
        position,
        coarse,
        band_half=band_half,
        patch_width=patch_width,
    )
    return EdgePrediction(
        edge=edge,
        coarse=float(coarse),
        refined=float(refined),
        offset=float(refined - coarse),
        confidence=confidence,
        entropy=entropy,
        peak_mass=peak_mass,
        canonical_position=position,
        tta_disagreement=tta_disagreement,
    )


def load_refiner(checkpoint_path: str | Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    config = dict(payload.get("config", {}))
    architecture = str(config.get("architecture", "v4")).lower()
    input_channels = int(config.get("input_channels", 7 if architecture == "v5" else 4))
    if architecture == "v5":
        model: nn.Module = EdgeRefinerV5(input_channels=input_channels)
    else:
        model = EdgeRefiner(input_channels=input_channels)
    model.load_state_dict(payload["model"])
    model.tta_height_flip = bool(config.get("tta_height_flip", architecture == "v5"))
    model.to(device).eval()
    return model, config
