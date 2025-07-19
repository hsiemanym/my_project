import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
import os

# ---- 설정 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device).eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2])

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ---- Generic penalty mask 불러오기 ----
semantic_mask = torch.load("./semantic_masks/semantic_mask_pdf.pt").to(device)  # [256]

# ---- Patch-wise feature 추출 함수 ----
def extract_patchwise_feature(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = feature_extractor(x)  # [1, 2048, 16, 16]
        feat = feat_map.flatten(2).squeeze(0).permute(1, 0)  # [256, 2048]
    return feat  # [256, 2048]

# ---- Test & Top1 이미지 경로 ----
test_img_path = "./dataset/test_test/test.png"
top1_img_path = "./dataset/Pretrain_Images/test_50/005.jpg"

# ---- 두 이미지의 masked feature 평균 계산 ----
feat_test = extract_patchwise_feature(test_img_path)        # [256, 2048]
feat_top1 = extract_patchwise_feature(top1_img_path)        # [256, 2048]

# Penalty 곱해주기 (비전형적인 patch 강조)
weighted_test = feat_test * semantic_mask.unsqueeze(1)      # [256, 2048]
weighted_top1 = feat_top1 * semantic_mask.unsqueeze(1)      # [256, 2048]

avg_feat_test = weighted_test.mean(dim=0)                   # [2048]
avg_feat_top1 = weighted_top1.mean(dim=0)

# ---- 최종 similarity 계산 ----
similarity = F.cosine_similarity(avg_feat_test.unsqueeze(0), avg_feat_top1.unsqueeze(0)).item()

print(f"\n🧠 Generic-penalty 기반 유사도 예측 결과:")
print(f"→ Cosine Similarity (비전형성 기반): {similarity:.4f}")
