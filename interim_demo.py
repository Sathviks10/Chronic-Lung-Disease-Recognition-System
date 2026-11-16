"""Small end-to-end inference demo for the interim project review."""

from pathlib import Path
import sys

try:
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from torchvision import transforms

    from model import get_model
except ModuleNotFoundError as exc:
    missing_package = exc.name or "a required package"
    raise SystemExit(
        f"Missing dependency: {missing_package}. "
        "Install the project dependencies from requirements.txt and try again."
    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_DIR / "sample_input.png"
CHECKPOINT_PATH = PROJECT_DIR / "lung_disease_model.pth"

# Keep the label order consistent with predict.py and data_loader.py.
CLASS_NAMES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def main():
    for path, description in (
        (IMAGE_PATH, "Sample input image"),
        (CHECKPOINT_PATH, "Model checkpoint"),
    ):
        if not path.is_file():
            raise SystemExit(f"{description} not found: {path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on {device}.")

    image = Image.open(IMAGE_PATH).convert("RGB")
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    model_input = preprocess(image).unsqueeze(0).to(device)

    model = get_model(num_classes=len(CLASS_NAMES))
    try:
        state_dict = torch.load(
            CHECKPOINT_PATH,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError, TypeError) as exc:
        raise SystemExit(
            "Could not load lung_disease_model.pth into the project's ResNet50 "
            f"architecture. Check that the checkpoint is a compatible model state dict.\n{exc}"
        ) from exc

    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(model_input)
        if logits.shape != (1, len(CLASS_NAMES)):
            raise SystemExit(
                f"Expected model output shape (1, {len(CLASS_NAMES)}), "
                f"but received {tuple(logits.shape)}."
            )
        probabilities = torch.sigmoid(logits)[0].cpu()

    print("\nDisease probabilities (sigmoid scores):")
    for name, score in zip(CLASS_NAMES, probabilities.tolist()):
        print(f"  {name:<20} {score:.4f} ({score:.1%})")

    top_scores, top_indices = torch.topk(probabilities, k=3)
    print("\nTop 3 predictions:")
    for rank, (score, index) in enumerate(zip(top_scores.tolist(), top_indices.tolist()), 1):
        print(f"  {rank}. {CLASS_NAMES[index]}: {score:.4f} ({score:.1%})")

    mean = torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
    std = torch.tensor(NORMALIZE_STD).view(3, 1, 1)
    processed_preview = (model_input[0].detach().cpu() * std + mean).clamp(0, 1)
    processed_preview = processed_preview.permute(1, 2, 0).numpy()

    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title(f"Original input: {IMAGE_PATH.name}")
    axes[1].imshow(processed_preview)
    axes[1].set_title("Processed model input (224 x 224)")
    for axis in axes:
        axis.axis("off")
    figure.suptitle("Chronic Lung Disease Detection — Interim Demo")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except OSError as exc:
        print(f"Could not read the image or model checkpoint: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
