
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Take the necessary parameters
def pred_and_plot_image(model: torch.nn.Module, class_names: List[str], target_image_path: str,
                        image_size: Tuple[int, int] = (224, 224), transform: torchvision.transforms = None,
                        device: torch.device = device):

    # 2. Open PIL Image
    img = Image.open(target_image_path)

    # 3. Create transforms
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std = [0.229, 0.224, 0.225])
        ])

    # 4. Model on target device
    model.to(device)

    # 5. model eval mode and inference context manager
    model.eval()
    with torch.inference_mode():
        # 6. Tranform the image and add a batch size dimension
        transformed_image =image_transform(img).unsqueeze(dim=0)

        # 7. Make the prediction
        target_image_pred = model(transformed_image.to(device))

    # 8. Logits to pred probs
    target_image_pred_probs = torch.softmax(target_image_pred, dim = 1)

    # 9. Pred probs to pred labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim = 1)

    # 10. Plot image with predicitons
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False);
