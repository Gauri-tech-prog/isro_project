import torch
import numpy as np
import cv2
import os
import requests

from PIL import Image, ImageEnhance
from torchvision import transforms
from model import GeneratorUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = GeneratorUNet(in_channels=1, out_channels=3)

# Paste your Hugging Face model download link here
MODEL_URL = "https://huggingface.co/gaurisakhale/ir2rgb-model/resolve/main/sat_generator_epoch100.pth"

MODEL_PATH = "model.pth"

# Download model only once
if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL)

    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

generator.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

generator.eval().to(device)

print(f"✅ Model loaded on {device}")


def sharpen_ir(pil_img):
    img_np = np.array(pil_img.convert('L'))

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8)
    )

    clahe_img = clahe.apply(img_np)

    blurred = cv2.GaussianBlur(
        clahe_img,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        clahe_img,
        1.2,
        blurred,
        -0.2,
        0
    )

    return Image.fromarray(sharpened).convert('L')


def super_resolve(pil_img):
    pil_img = pil_img.convert('L')

    w, h = pil_img.size

    upscaled = pil_img.resize(
        (w * 2, h * 2),
        Image.LANCZOS
    )

    enhancer = ImageEnhance.Sharpness(upscaled)

    return enhancer.enhance(1.5).convert('L')


def colorize(pil_img, debug=False):

    gray = pil_img.convert('L').resize((256, 256))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    x = transform(gray).unsqueeze(0).to(device)

    with torch.no_grad():
        raw_output = generator(x)

    if debug:
        print(
            f"Raw generator output — "
            f"min: {raw_output.min().item():.3f}, "
            f"max: {raw_output.max().item():.3f}, "
            f"mean: {raw_output.mean().item():.3f}"
        )

    fake_rgb = (raw_output * 0.5 + 0.5).clamp(0, 1)

    fake_np = (
        fake_rgb.squeeze(0)
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    rgb_img = (fake_np * 255).astype(np.uint8)

    return Image.fromarray(rgb_img)


def correct_colors(pil_img):

    img = np.array(pil_img).astype(np.float32)

    result = np.zeros_like(img)

    for c in range(3):

        channel = img[:, :, c]

        p2, p98 = np.percentile(channel, (2, 98))

        if p98 - p2 < 1:
            result[:, :, c] = channel
            continue

        result[:, :, c] = np.clip(
            (channel - p2) / (p98 - p2) * 255.0,
            0,
            255
        )

    return Image.fromarray(result.astype(np.uint8))


def full_pipeline(pil_img, debug=False):

    sharpened = sharpen_ir(pil_img)

    sr_img = super_resolve(sharpened)

    rgb_out = colorize(
        pil_img,
        debug=debug
    )

    rgb_out = correct_colors(rgb_out)

    final = rgb_out.resize(
        (512, 512),
        Image.LANCZOS
    )

    return sharpened, sr_img.resize((512, 512)), final

