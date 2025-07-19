import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ---- 기본 설정 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True).to(device).eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- Hook ----
features, gradients = None, None

def save_activation(module, input, output):
    global features
    features = output.detach()

def save_gradient(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0].detach()

target_layer = model.layer4
target_layer.register_forward_hook(save_activation)
target_layer.register_backward_hook(save_gradient)

# ---- Grad-CAM++ 계산 함수 ----
def compute_gradcampp(image_path, penalty_path):
    global features, gradients
    features, gradients = None, None

    # 이미지 로드
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    # penalty mask
    penalty = torch.load(penalty_path).to(device)  # [256]
    penalty_mask = penalty.view(1, 1, 16, 16)

    # Forward & Backward
    model.zero_grad()
    output = model(x)
    pred_class = output.argmax(dim=1)
    score = output[0, pred_class]
    score.backward(retain_graph=True)

    grad = gradients           # [1, C, H, W]
    feat = features            # [1, C, H, W]
    B, C, H, W = grad.shape

    # Grad-CAM++ weights 계산
    grad_2 = grad ** 2
    grad_3 = grad ** 3
    eps = 1e-8
    sum_grad = torch.sum(grad, dim=(2, 3), keepdim=True) + eps
    alpha_num = grad_2
    alpha_denom = 2 * grad_2 + grad_3 * sum_grad
    alpha = alpha_num / (alpha_denom + eps)  # [1, C, H, W]
    weights = torch.sum(alpha * F.relu(grad), dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

    # penalty 적용
    feat = feat * penalty_mask
    weights = weights * penalty_mask

    # CAM 계산
    cam = F.relu(torch.sum(weights * feat, dim=1, keepdim=True))  # [1, 1, H, W]
    cam = F.interpolate(cam, size=(512, 512), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam, np.array(img.resize((512, 512))) / 255.0

# ---- 이미지 경로 ----
test_path = "./dataset/test_test/test.png"
top1_path = "./dataset/Pretrain_Images/test_50/005.jpg"
penalty_path = "semantic_masks/test_patch_penalty.pt"

# ---- Grad-CAM++ 실행 ----
test_cam, test_img_np = compute_gradcampp(test_path, penalty_path)
top1_cam, top1_img_np = compute_gradcampp(top1_path, penalty_path)

# ---- Counterfactual 계산 ----
counter_test = np.abs(test_cam - top1_cam)
counter_top1 = np.abs(top1_cam - test_cam)

# ---- 시각화 도우미 ----
def overlay(img, cam, alpha=0.45, cmap='jet'):
    cmap = plt.get_cmap(cmap)
    heatmap = cmap(cam)[..., :3]
    return heatmap * alpha + img * (1 - alpha)

# ---- 4장 시각화 ----
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(test_img_np)
axs[0, 0].set_title("Test Original")
axs[0, 1].imshow(overlay(test_img_np, test_cam))
axs[0, 1].set_title("Test Factual (Grad-CAM++)")
axs[0, 2].imshow(overlay(test_img_np, counter_test))
axs[0, 2].set_title("Test Counterfactual")

axs[1, 0].imshow(top1_img_np)
axs[1, 0].set_title("Top-1 Original")
axs[1, 1].imshow(overlay(top1_img_np, top1_cam))
axs[1, 1].set_title("Top-1 Factual (Grad-CAM++)")
axs[1, 2].imshow(overlay(top1_img_np, counter_top1))
axs[1, 2].set_title("Top-1 Counterfactual")

for ax in axs.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
