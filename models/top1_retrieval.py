import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# --------- 설정 ---------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device).eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2])

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# --------- 함수 ---------
def extract_avg_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = feature_extractor(x)  # [1, 2048, 16, 16]
        feat = feat_map.flatten(2).squeeze(0).permute(1, 0)  # [256, 2048]
        avg_feat = feat.mean(dim=0)  # [2048]
    return avg_feat.cpu()

# --------- 경로 ---------
test_img_path = "./dataset/test_test/test.png"
gallery_dir = "./dataset/Pretrain_Images/test_50/"

# --------- Test 이미지 임베딩 ---------
query_feat = extract_avg_feature(test_img_path)

# --------- 전체 gallery 이미지들과 유사도 계산 ---------
similarities = []
gallery_list = sorted(os.listdir(gallery_dir))

for fname in tqdm(gallery_list):
    if not fname.endswith((".jpg", ".jpeg", ".png")): continue
    path = os.path.join(gallery_dir, fname)
    feat = extract_avg_feature(path)
    sim = F.cosine_similarity(query_feat.unsqueeze(0), feat.unsqueeze(0)).item()
    similarities.append((fname, sim))

# --------- 가장 유사한 이미지 출력 ---------
similarities.sort(key=lambda x: x[1], reverse=True)
top1 = similarities[0]

print(f"\n📌 Top-1 가장 유사한 이미지: {top1[0]}")
print(f"→ Cosine similarity: {top1[1]:.4f}")
