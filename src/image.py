from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from . import io
from . import processing
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class DiskGeometry:
    cx: float
    cy: float
    r: float


class SolarImage:
    """
    Representation of a solar image.
    """

    def __init__(self, data, header=None):
        self.original = data.copy()
        self.data = data.copy()
        self.header = header
        self.history = []
        self.disk: DiskGeometry | None = None

    @classmethod
    def from_path(cls, path: str, grayscale: bool = True) -> "SolarImage":
        """
        Create SolarImage from a file path. Chooses loader based on extension.
        """
        suffix = Path(path).suffix.lower()

        if suffix in {".fits", ".fit", ".fts"}:
            data, header = io.load_fits(path)  # returns tuple
            return cls(data, header)

        # common image formats
        if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}:
            data = io.load_image(path, grayscale=grayscale)
            return cls(data, header=None)

        raise ValueError(f"Unsupported file extension: {suffix}")

    def reset(self):
        self.data = self.original.copy()
        self.history.append("reset")

    def copy(self):
        c = SolarImage(self.data.copy(), self.header)
        c.history = self.history.copy()
        c.disk = self.disk
        return c

    def show(self, title: str = "Solar Image", cmap: str = "gray"):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.data, cmap=cmap)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def set_disk(self, cx: float, cy: float, r: float):
        self.disk = DiskGeometry(float(cx), float(cy), float(r))
        self.history.append(f"set_disk(cx={cx},cy={cy},r={r})")
        return self

    def set_disk_from_header(self):
        """
        Extract disk geometry from FITS header using INTI_* keys only.
        Required keys:
          - INTI_XC (center x in pixels)
          - INTI_YC (center y in pixels)
          - INTI_R  (radius in pixels)
        """
        if self.header is None:
            raise ValueError("No FITS header available.")

        h = self.header

        cx = h.get("INTI_XC")
        cy = h.get("INTI_YC")
        r = h.get("INTI_R")

        if cx is None or cy is None or r is None:
            keys = list(h.keys()) if hasattr(h, "keys") else []
            raise KeyError(
                "Missing required FITS header keys. "
                "Need INTI_XC, INTI_YC, INTI_R. "
                f"Sample keys: {keys[:60]}"
            )

        return self.set_disk(float(cx), float(cy), float(r))


    def set_disk_from(self, other: "SolarImage"):
        """
        Copy disk geometry from another SolarImage.

        If image sizes differ, scale geometry based on width/height ratios.
        This assumes the images are the same scene and only resized uniformly.
        """
        if other.disk is None:
            raise ValueError("Source image has no disk geometry set.")

        src_h, src_w = other.data.shape[:2]
        dst_h, dst_w = self.data.shape[:2]

        sx = dst_w / src_w
        sy = dst_h / src_h

        # If non-uniform scaling happened, this is an approximation.
        s = (sx + sy) / 2.0

        cx = other.disk.cx * sx
        cy = other.disk.cy * sy
        r = other.disk.r * s

        self.disk = DiskGeometry(cx, cy, r)
        self.history.append("set_disk_from(other)")
        return self

    # ================
    # Image Processing
    # ================
    def otsu(self):
        """
        Apply Otsu thresholding.
        """
        self.data = processing.otsu_threshold(self.data)
        self.history.append("otsu")
        return self

    def prominences_otsu(self, inner: int = 5, outer: int = 150, tophat_kernel: int = 31):
        if self.disk is None:
            raise ValueError("Disk geometry not set.")

        cx, cy, r = self.disk.cx, self.disk.cy, self.disk.r

        work = self.data
        if tophat_kernel and tophat_kernel > 0:
            work = processing.white_tophat(work, kernel_size=tophat_kernel)
            self.history.append(f"white_tophat(k={tophat_kernel})")

        mask = processing.ring_mask(work.shape[:2], cx, cy, r, inner_margin=inner, outer_margin=outer)
        self.data = processing.apply_otsu_on_mask(work, mask)
        self.history.append(f"prominences_otsu(inner={inner},outer={outer})")
        return self

    def calc_protuberanzenprofil(self, start_offset: float = 5.0, max_length: float = 250.0, step: float = 1.0) -> list[
        float]:
        if self.disk is None:
            raise ValueError("Disk geometry not set.")
        cx, cy, r = self.disk.cx, self.disk.cy, self.disk.r

        # assumes self.data is a binary mask (0/255)
        prof = processing.calc_protuberanzenprofil(
            self.data, cx, cy, r,
            start_offset=start_offset,
            max_length=max_length,
            step=step,
            white_value=255,
        )
        self.history.append(f"calc_protuberanzenprofil(start={start_offset},max={max_length},step={step})")
        return prof.tolist()
