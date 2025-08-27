from PIL import Image, ImageFilter
import numpy as np

def _try_cv2():
    try:
        import cv2  # type: ignore
        return cv2
    except Exception:
        return None

def postprocess_image(img: Image.Image, kind: str,
                      strength: str = "auto", window: int = 3,
                      allow_cv2: bool = True) -> Image.Image:
    kind = kind.lower()
    strength = (strength or "auto").lower()
    window = max(3, int(window) | 1)  # 奇数

    if kind == "depth":
        im = img.convert("L")
        if strength in ("auto", "light"):
            return im.filter(ImageFilter.MedianFilter(size=window))
        elif strength == "medium":
            return im.filter(ImageFilter.MedianFilter(size=window)).filter(ImageFilter.MedianFilter(size=window))
        else:  # strong
            if allow_cv2 and (cv2 := _try_cv2()) is not None:
                arr = np.array(im, dtype=np.uint8)
                d = max(5, window)
                out = cv2.bilateralFilter(arr, d=d, sigmaColor=30, sigmaSpace=7)
                return Image.fromarray(out, mode="L")
            return im.filter(ImageFilter.MedianFilter(size=window)).filter(ImageFilter.MedianFilter(size=window))

    if kind == "edge":
        im = img.convert("L").filter(ImageFilter.MedianFilter(size=window))
        # opening
        im = im.filter(ImageFilter.MinFilter(size=3)).filter(ImageFilter.MaxFilter(size=3))
        if strength in ("medium", "strong"):
            # closing
            im = im.filter(ImageFilter.MaxFilter(size=3)).filter(ImageFilter.MinFilter(size=3))
        return im.point(lambda p: 255 if p >= 128 else 0).convert("1")

    if kind == "seg":
        im = img.convert("RGB")
        if strength in ("auto", "light"):
            return im.filter(ImageFilter.ModeFilter(size=window))
        elif strength == "medium":
            return im.filter(ImageFilter.ModeFilter(size=window)).filter(ImageFilter.ModeFilter(size=window))
        else:  # strong
            im = im.filter(ImageFilter.ModeFilter(size=window))
            im = im.filter(ImageFilter.MedianFilter(size=max(3, window)))
            return im.filter(ImageFilter.ModeFilter(size=window))

    return img
