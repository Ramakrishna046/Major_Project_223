
import json
import torch
from PIL import Image
from torchvision import models, transforms

from core.config import (
    CLASSIFICATION_MODEL,
    CLASS_NAMES_FILE,
    RECYCLABILITY_FILE,
    DEVICE,
)


class WasteClassifier:

    def __init__(self):
        self.device = torch.device(DEVICE)

        with open(CLASS_NAMES_FILE, "r", encoding="utf-8") as file:
            self.class_names = json.load(file)

        with open(RECYCLABILITY_FILE, "r", encoding="utf-8") as file:
            self.recyclability_mapping = json.load(file)

        # Recreate the trained torchvision EfficientNet-B0 architecture
        self.model = models.efficientnet_b0(weights=None)

        self.model.classifier[1] = torch.nn.Linear(
            self.model.classifier[1].in_features,
            len(self.class_names),
        )

        checkpoint = torch.load(
            CLASSIFICATION_MODEL,
            map_location=self.device,
            weights_only=True,
        )

        # Load the trained model weights
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Remove DataParallel prefixes if present
        state_dict = {
            key.replace("module.", "", 1): value
            for key, value in state_dict.items()
        }

        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Match the training/validation preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def classify(self, image):
        """
        Accepts an OpenCV BGR image or a PIL image.
        """

        if image is None:
            raise ValueError("Input image is empty.")

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image[:, :, ::-1])

        image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.inference_mode():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities, dim=1
        )

        category = self.class_names[predicted_index.item()]

        return {
            "category": category,
            "confidence": round(confidence.item() * 100, 2),
            "recyclability": self.recyclability_mapping.get(
                category, "Unknown"
            ),
        }
