import io
import cv2
import numpy as np
from PIL import Image


class FacePreprocessor:
    def __init__(self) -> None:
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"  # type: ignore[attr-defined]
        )

    def preprocess(self, image_bytes: bytes, img_size: int) -> np.ndarray:
        with Image.open(io.BytesIO(image_bytes)) as image:
            rgb_image = image.convert("RGB")
        face_square = self._crop_face_square(rgb_image)
        gray = face_square.convert("L")
        resized = gray.resize((img_size, img_size))
        arr = np.asarray(resized, dtype=np.float32)
        return np.expand_dims(arr, axis=(0, -1))

    def _crop_face_square(self, rgb_image: Image.Image) -> Image.Image:
        gray_for_detection = np.asarray(rgb_image.convert("L"), dtype=np.uint8)
        faces = self._cascade.detectMultiScale(
            gray_for_detection,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )

        width, height = rgb_image.size
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
            side = max(w, h)
            center_x = x + (w // 2)
            center_y = y + (h // 2)
            left = max(0, center_x - (side // 2))
            top = max(0, center_y - (side // 2))
            right = min(width, left + side)
            bottom = min(height, top + side)
            left = max(0, right - side)
            top = max(0, bottom - side)
            return rgb_image.crop((left, top, right, bottom))

        side = min(width, height)
        left = (width - side) // 2
        top = (height - side) // 2
        return rgb_image.crop((left, top, left + side, top + side))
