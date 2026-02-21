import cv2
import numpy as np


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image to 0..255 uint8."""
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return img.astype(np.uint8)


def disk_mask(shape: tuple[int, int], cx: float, cy: float, r: float, margin: int = 0) -> np.ndarray:
    """
    Create boolean mask for pixels inside solar disk (with optional margin).
    """
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    rr = (xx - cx) ** 2 + (yy - cy) ** 2
    return rr <= (r + margin) ** 2


def ring_mask(shape: tuple[int, int], cx: float, cy: float, r: float, inner_margin: int = 0, outer_margin: int = 80) -> np.ndarray:
    """
    Mask for an annulus outside the limb:
    (r+inner_margin) < radius <= (r+outer_margin)
    """
    h, w = shape
    yy, xx = np.ogrid[:h, :w]
    rr = (xx - cx) ** 2 + (yy - cy) ** 2
    inner = (r + inner_margin) ** 2
    outer = (r + outer_margin) ** 2
    return (rr > inner) & (rr <= outer)


def apply_otsu_on_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply Otsu thresholding using only pixels where mask==True.
    Returns binary image (0/255) where only mask region can become white.
    """
    img_u8 = to_uint8(image)
    vals = img_u8[mask]
    if vals.size < 1000:
        raise RuntimeError("Masked region too small for Otsu.")

    # compute global Otsu threshold on whole image, but this is influenced by disk
    # better: compute threshold using masked pixels (1D), then apply to mask only
    # Otsu for 1D: use cv2.threshold on a column vector
    v = vals.reshape(-1, 1)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    out = np.zeros_like(img_u8, dtype=np.uint8)
    out[mask] = (img_u8[mask] > t).astype(np.uint8) * 255
    return out


def white_tophat(image: np.ndarray, kernel_size: int = 31) -> np.ndarray:
    """
    Enhance small bright structures on darker background.
    """
    img_u8 = to_uint8(image)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(img_u8, cv2.MORPH_TOPHAT, k)


def otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu thresholding.
    """
    # ensure uint8
    if image.dtype != np.uint8:
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype(np.uint8)
    else:
        img_uint8 = image

    _, thresh = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return thresh


def bilateral_filter(image: np.ndarray,
                     d: int = 5,
                     sigma_color: float = 150,
                     sigma_space: float = 150) -> np.ndarray:
    """
    applies bilateral filtering to reduce noise without erasing important edges
    """
    if image.dtype != np.uint8:
        img_norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = img_norm.astype(np.uint8)
    else:
        img_uint8 = image
    return cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)


def calc_protuberanzenprofil(
    binary: np.ndarray,
    cx: float,
    cy: float,
    r: float,
    start_offset: float = 5.0,
    max_length: float = 250.0,
    step: float = 1.0,
    white_value: int = 255,
) -> np.ndarray:
    """
    Compute prominence radial extent (in pixels) for each degree 0..359.

    Angle convention:
      - 0° = north (up)
      - increases clockwise
      - 90° = east (right)
      - 180° = south (down)
      - 270° = west (left)

    Parameters
    ----------
    binary : np.ndarray
        Binary mask image (0/255 or 0/1). Prominences should be white.
    cx, cy, r : float
        Disk center and radius in pixels.
    start_offset : float
        Start sampling at radius r + start_offset (avoid limb/ring artifacts).
    max_length : float
        Maximum prominence length to search outward (pixels).
    step : float
        Sampling step along the ray in pixels.
    white_value : int
        Value considered as "white" (default 255). If your mask is 0/1, set to 1.

    Returns
    -------
    np.ndarray
        Array of length 360, where entry[i] is prominence length (pixels) at angle i.
        0 means nothing detected.
    """
    if binary.ndim != 2:
        raise ValueError("binary mask must be a 2D array")

    h, w = binary.shape
    prof = np.zeros(360, dtype=np.float32)

    # radii sampled outward from the limb
    radii = np.arange(r + start_offset, r + start_offset + max_length + 1e-6, step, dtype=np.float32)

    # precompute for speed
    cx_f = float(cx)
    cy_f = float(cy)

    for deg in range(360):
        theta = np.deg2rad(deg)

        # 0° = up, clockwise
        dx = np.sin(theta)
        dy = -np.cos(theta)

        xs = cx_f + radii * dx
        ys = cy_f + radii * dy

        xi = np.rint(xs).astype(np.int32)
        yi = np.rint(ys).astype(np.int32)

        # keep only points inside image bounds
        inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if not np.any(inside):
            prof[deg] = 0.0
            continue

        xi = xi[inside]
        yi = yi[inside]
        rr = radii[inside]

        vals = binary[yi, xi]
        hits = (vals == white_value)

        if not np.any(hits):
            prof[deg] = 0.0
            continue

        # length = farthest hit distance measured from the start radius
        last_idx = np.flatnonzero(hits)[-1]
        prof[deg] = rr[last_idx] - (r + start_offset)

    return prof

