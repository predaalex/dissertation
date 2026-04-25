from __future__ import annotations

from io import BytesIO
from pathlib import Path

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
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_config: dict | None = None
        self.eval_transform = self._build_eval_transform()

    def healthcheck(self) -> bool:
        if not self.checkpoint_path.exists():
            return False
        try:
            self._ensure_model_loaded()
            return True
        except EmotionServiceError:
            return False

    def predict(self, image_bytes: bytes) -> dict:
        self._ensure_model_loaded()

        try:
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
