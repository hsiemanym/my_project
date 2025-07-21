import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

# 경로 설정
test_path = "./dataset/test_test/test.png"
top1_path = "./dataset/Pretrain_Images/test_50/005.jpg"
test_penalty_path = "./semantic_masks/test_patch_penalty_layer3.pt"
top1_penalty_path = "./semantic_masks/top1_patch_penalty_layer3.pt"

# 전처리
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Feature 추출기 정의
resnet = models.resnet50(pretrained=True)
layer_outputs = {}

def hook_fn(module, input, output):
    layer_outputs["features"] = output

resnet.layer3.register_forward_hook(hook_fn)
resnet.eval()

# 이미지 로드 및 feature 추출
def get_features(path):
    img = Image.open(path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        _ = resnet(tensor)
    feat = layer_outputs["features"]  # [1, C, H, W]
    feat = F.adaptive_avg_pool2d(feat, (32, 32))
    feat = feat.squeeze(0).permute(1, 2, 0).reshape(-1, 1024)  # [1024, 1024]
    return F.normalize(feat, dim=1)

# 유사도 계산 함수
def compute_similarity(feat1, feat2, penalty1, penalty2):
    sim = torch.matmul(feat1, feat2.T)  # [1024, 1024]
    sim = sim.mean(dim=1)  # [1024] 기준: test 이미지 patch마다 평균 유사도

    # penalty → weight 계산
    weight = (1 - penalty1) * (1 - penalty2)  # [1024]
    weighted_sim = sim * weight
    return weighted_sim.mean().item()

# 실행
feat_test = get_features(test_path)
feat_top1 = get_features(top1_path)

penalty_test = torch.load(test_penalty_path)  # [1024]
penalty_top1 = torch.load(top1_penalty_path)  # [1024]

score = compute_similarity(feat_test, feat_top1, penalty_test, penalty_top1)
print(f"✅ Generic-aware Similarity Score: {score:.4f}")
