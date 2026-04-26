from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


CLASS_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


class EmotionServiceError(RuntimeError):
    """Raised when the emotion classifier cannot be used."""


class EmotionService:
    def __init__(self, checkpoint_path: str, face_cropping_enabled: bool = True):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_cropping_enabled = face_cropping_enabled
        self.model = None
        self.model_config: dict | None = None
        self.eval_transform = self._build_eval_transform()
        self.face_cascade = self._load_face_cascade()

    def healthcheck(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        try:
            self._ensure_model_loaded()
            return True
        except EmotionServiceError:
            return False

    def predict(self, image_bytes: bytes, disable_face_cropping: bool = False) -> dict:
        self._ensure_model_loaded()

        crop_metadata = {
            "face_detected": False,
            "face_crop_applied": False,
            "face_bbox": None,
        }

        if self.face_cropping_enabled and not disable_face_cropping:
            image, crop_metadata = self._load_face_cropped_image(image_bytes)
        else:
            image = None

        try:
            if image is None:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            raise EmotionServiceError(f"Invalid image input: {exc}") from exc

        image_tensor = self.eval_transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

        predicted_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_index].item())
        probability_map = {
            CLASS_NAMES[idx]: float(probabilities[idx].item())
            for idx in range(len(CLASS_NAMES))
        }

        return {
            "emotion_label": CLASS_NAMES[predicted_index],
            "emotion_index": predicted_index,
            "confidence": confidence,
            "probabilities": probability_map,
            **crop_metadata,
        }

    def _ensure_model_loaded(self) -> None:
        if self.model is not None:
            return

        if not self.checkpoint_path.exists():
            raise EmotionServiceError(
                f"Emotion model checkpoint not found at {self.checkpoint_path}."
            )

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception as exc:  # noqa: BLE001
            raise EmotionServiceError(f"Failed to load checkpoint: {exc}") from exc

        config = checkpoint.get("config", {})
        num_classes = int(config.get("num_classes", len(CLASS_NAMES)))
        dropout = float(config.get("dropout", 0.2))

        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        in_features = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features, num_classes),
        )

        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as exc:  # noqa: BLE001
            raise EmotionServiceError(f"Checkpoint weights do not match the model: {exc}") from exc

        self.model = model.to(self.device)
        self.model.eval()
        self.model_config = config

    @staticmethod
    def _build_eval_transform():
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        return transforms.Compose(
            [
                transforms.Resize((232, 232)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    @staticmethod
    def _load_face_cascade():
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        if cascade.empty():
            return None
        return cascade

    def _load_face_cropped_image(self, image_bytes: bytes) -> tuple[Image.Image | None, dict]:
        if self.face_cascade is None:
            return None, {
                "face_detected": False,
                "face_crop_applied": False,
                "face_bbox": None,
            }

        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        bgr_image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
        if bgr_image is None:
            return None, {
                "face_detected": False,
                "face_crop_applied": False,
                "face_bbox": None,
            }

        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
        )

        if len(faces) == 0:
            return None, {
                "face_detected": False,
                "face_crop_applied": False,
                "face_bbox": None,
            }

        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        padded_bbox = self._pad_bbox(x, y, w, h, bgr_image.shape[1], bgr_image.shape[0])
        x1, y1, x2, y2 = padded_bbox
        cropped_bgr = bgr_image[y1:y2, x1:x2]
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        cropped_image = Image.fromarray(cropped_rgb)

        return cropped_image, {
            "face_detected": True,
            "face_crop_applied": True,
            "face_bbox": [int(x1), int(y1), int(x2), int(y2)],
        }

    @staticmethod
    def _pad_bbox(x: int, y: int, w: int, h: int, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(image_width, x + w + pad_x)
        y2 = min(image_height, y + h + pad_y)
        return x1, y1, x2, y2
