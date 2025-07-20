import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import cv2
from PIL import Image

# ─── 로딩 유틸 ─── #
def load_patch_features(img_path, feat_path):
    img = Image.open(img_path).convert("RGB").resize((512, 512))
    feat = torch.load(feat_path)
    feat = F.normalize(feat, dim=1)
    return img, feat

# ─── Grad-CAM++ 스타일 유사도 기반 히트맵 계산 ─── #
def compute_patch_similarity_map(source_feat, target_feat, penalty):
    sim = F.cosine_similarity(source_feat, target_feat, dim=1)  # [256]
    weight = 1 - penalty                                        # [256]
    sim_weighted = sim * weight
    return sim, sim_weighted

# ─── 16x16 → 512x512 부드러운 업샘플링 ─── #
def upscale_heatmap(heatmap_16, target_size=(512, 512)):
    heatmap = heatmap_16.view(16, 16).cpu().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_resized = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    return Image.fromarray(heatmap_colored)

# ─── 이미지에 히트맵 합성 ─── #
def overlay_heatmap(img, heatmap, alpha=0.5):
    img = img.convert("RGB").resize((512, 512))
    return Image.blend(img, heatmap, alpha)

# ─── 전체 시각화 ─── #
def visualize_4_heatmaps(test_img_path, top1_img_path,
                         test_feat_path, top1_feat_path,
                         penalty_path):

    penalty = torch.load(penalty_path)  # [256]
    test_img, test_feat = load_patch_features(test_img_path, test_feat_path)
    top1_img, top1_feat = load_patch_features(top1_img_path, top1_feat_path)

    sim, sim_weighted = compute_patch_similarity_map(test_feat, top1_feat, penalty)

    # Grad-CAM++ 스타일 heatmap
    test_factual_hm = upscale_heatmap(sim_weighted)
    test_counter_hm = upscale_heatmap(1 - sim)
    top1_factual_hm = upscale_heatmap(sim_weighted)
    top1_counter_hm = upscale_heatmap(1 - sim)

    # 이미지와 합성
    test_factual = overlay_heatmap(test_img, test_factual_hm)
    test_counter = overlay_heatmap(test_img, test_counter_hm)
    top1_factual = overlay_heatmap(top1_img, top1_factual_hm)
    top1_counter = overlay_heatmap(top1_img, top1_counter_hm)

    # 시각화
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs = axs.flatten()
    axs[0].imshow(test_factual); axs[0].set_title("Test – Factual")
    axs[1].imshow(test_counter); axs[1].set_title("Test – Counterfactual")
    axs[2].imshow(top1_factual); axs[2].set_title("Top-1 – Factual")
    axs[3].imshow(top1_counter); axs[3].set_title("Top-1 – Counterfactual")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.show()


visualize_4_heatmaps(
    test_img_path="dataset/test_test/test.png",
    top1_img_path="dataset/Pretrain_Images/test_50/005.jpg",
    test_feat_path="dataset/test_test/test_patch.pt",
    top1_feat_path="dataset/test_test/top1_patch.pt",
    penalty_path="semantic_masks/test_patch_penalty_cdf.pt"
)
