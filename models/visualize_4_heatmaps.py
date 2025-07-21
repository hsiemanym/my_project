import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 경로 설정
TEST_IMG_PATH = "./dataset/test_test/test.png"
TOP1_IMG_PATH = "./dataset/Pretrain_Images/test_50/005.jpg"
TEST_PENALTY_PATH = "./semantic_masks/test_patch_penalty_layer3.pt"
TOP1_PENALTY_PATH = "./semantic_masks/top1_patch_penalty_layer3.pt"
SAVE_PATH = "./heatmaps_layer4.png"

# 모델 불러오기
resnet = models.resnet50(pretrained=True)
resnet.eval()

# 이미지 로드 함수
def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Grad-CAM 생성 함수
def get_gradcam_map(model, img_tensor, target_layer):
    activations = []
    gradients = []

    def fwd_hook(module, input, output):
        activations.append(output)

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_backward_hook(bwd_hook)

    output = model(img_tensor)
    class_idx = torch.argmax(output)
    score = output[:, class_idx]
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    grads = gradients[0]  # [1, C, H, W]
    acts = activations[0]  # [1, C, H, W]

    weights = grads.mean(dim=(2, 3), keepdim=True)  # GAP
    cam = (weights * acts).sum(dim=1).squeeze()  # [H, W]
    cam = torch.relu(cam)

    # 📌 percentile 기반 정규화
    cam_np = cam.detach().cpu().numpy()
    p_min, p_max = np.percentile(cam_np, 5), np.percentile(cam_np, 95)
    cam_np = np.clip(cam_np, p_min, p_max)
    cam_np = (cam_np - p_min) / (p_max - p_min + 1e-8)

    # ✅ 여기가 핵심!
    cam_np = cv2.resize(cam_np, (512, 512))  # 512x512로 업샘플링

    return cam_np  # [512, 512] numpy


# Heatmap 오버레이
def overlay_heatmap(cam, img_tensor):
    cam = cv2.resize(cam, (512, 512))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = Image.fromarray(heatmap).convert("RGBA")

    orig = img_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    orig = (orig * 255).astype(np.uint8)
    orig_img = Image.fromarray(orig).convert("RGBA")

    return Image.blend(orig_img, heatmap, alpha=0.5)

# Penalty 적용
# Counterfactual 생성 시 penalty resize 추가
def apply_penalty(cam, penalty):
    penalty = torch.clamp(penalty, 0, 1).cpu().numpy()
    penalty_resized = cv2.resize(penalty, (512, 512))  # ✅ 업샘플링
    return cam * (1 - penalty_resized)

# 로드
test_img = load_image(TEST_IMG_PATH)
top1_img = load_image(TOP1_IMG_PATH)
test_penalty = torch.load(TEST_PENALTY_PATH).view(32, 32)
top1_penalty = torch.load(TOP1_PENALTY_PATH).view(32, 32)

# CAM 생성
cam_test = get_gradcam_map(resnet, test_img, resnet.layer4)
cam_top1 = get_gradcam_map(resnet, top1_img, resnet.layer4)

# Counterfactual
cam_test_cf = apply_penalty(cam_test, test_penalty)
cam_top1_cf = apply_penalty(cam_top1, top1_penalty)

# 시각화
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
titles = ['Test - Factual', 'Test - Counterfactual', 'Top-1 - Factual', 'Top-1 - Counterfactual']
images = [test_img, test_img, top1_img, top1_img]
cams = [cam_test, cam_test_cf, cam_top1, cam_top1_cf]

for ax, title, img, cam in zip(axs.flatten(), titles, images, cams):
    heatmap_img = overlay_heatmap(cam, img)
    ax.imshow(heatmap_img)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig(SAVE_PATH)
print(f"✅ 저장 완료: {SAVE_PATH}")
