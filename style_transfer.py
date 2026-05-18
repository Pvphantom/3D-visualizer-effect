import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


CONTENT_LAYERS = ["conv_4"]
STYLE_LAYERS = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(path: str, max_size: int = 512, device: torch.device = None) -> torch.Tensor:
    """Load an image, resize to max_size, and convert to a normalized tensor."""
    image = Image.open(path).convert("RGB")
    scale = max_size / max(image.size)
    new_size = (int(image.size[0] * scale), int(image.size[1] * scale))
    image = image.resize(new_size, Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    tensor = transform(image).unsqueeze(0)
    return tensor.to(device) if device else tensor


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor back to a PIL Image."""
    img = tensor.clone().detach().cpu().squeeze(0)
    for i, (m, s) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD)):
        img[i] = img[i] * s + m
    img = img.clamp(0, 1)
    return transforms.ToPILImage()(img)


def _gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class ContentLoss(nn.Module):
    def __init__(self, target: torch.Tensor):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x


class StyleLoss(nn.Module):
    def __init__(self, target_feature: torch.Tensor):
        super().__init__()
        self.target = _gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0)

    def forward(self, x):
        self.loss = F.mse_loss(_gram_matrix(x), self.target)
        return x


def _build_model(content_img: torch.Tensor, style_img: torch.Tensor, device: torch.device):
    """Build the style transfer model by inserting loss modules after VGG conv layers."""
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    model = nn.Sequential()
    content_losses = []
    style_losses = []

    conv_i = 0
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_i += 1
            name = f"conv_{conv_i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{conv_i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_i}"
        else:
            name = f"other_{conv_i}"

        model.add_module(name, layer)

        if name in CONTENT_LAYERS:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{conv_i}", cl)
            content_losses.append(cl)

        if name in STYLE_LAYERS:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f"style_loss_{conv_i}", sl)
            style_losses.append(sl)

    last_loss_idx = 0
    for i, (name, _) in enumerate(model.named_children()):
        if "content_loss" in name or "style_loss" in name:
            last_loss_idx = i
    model = model[:last_loss_idx + 1]

    return model, content_losses, style_losses


def run_style_transfer(
    content_path: str,
    style_path: str,
    num_steps: int = 300,
    content_weight: float = 1.0,
    style_weight: float = 1e6,
    max_size: int = 512,
    on_step=None,
) -> Image.Image:
    """Run Gatys et al. neural style transfer.

    on_step: optional callback(step, total_steps, pastiche_tensor) called periodically.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    content_img = load_image(content_path, max_size, device)
    style_img = load_image(style_path, max_size, device)

    if style_img.shape[2:] != content_img.shape[2:]:
        style_img = F.interpolate(style_img, size=content_img.shape[2:], mode="bilinear", align_corners=False)

    model, content_losses, style_losses = _build_model(content_img, style_img, device)

    pastiche = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([pastiche])

    step_count = [0]

    def closure():
        pastiche.data.clamp_(
            min=-max(m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)),
            max=max((1 - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)),
        )
        optimizer.zero_grad()
        model(pastiche)

        c_loss = sum(cl.loss for cl in content_losses) * content_weight
        s_loss = sum(sl.loss for sl in style_losses) * style_weight
        total = c_loss + s_loss
        total.backward()

        step_count[0] += 1
        if step_count[0] % 50 == 0 or step_count[0] == 1:
            print(f"  Step {step_count[0]:>4d}/{num_steps}  content={c_loss.item():.2f}  style={s_loss.item():.4f}")
            if on_step:
                on_step(step_count[0], num_steps, pastiche)

        return total

    for _ in range(num_steps):
        optimizer.step(closure)
        if step_count[0] >= num_steps:
            break

    return tensor_to_image(pastiche)
