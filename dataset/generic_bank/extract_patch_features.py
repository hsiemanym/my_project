# 설명

# GPU 설정
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # line5

# 경로 설정
input_dir = "/Users/kimhoeyeon/Desktop/my_project/dataset/Pretrain_Images/test_50/"
output_dir = "./patch_features"
os.makedirs(output_dir, exist_ok=True)

# 512x512용 전처리 정의
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # ✅ 512 입력
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ResNet50에서 layer4까지 불러오는 모듈
class ResNet50_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-2])  # conv1 ~ layer4

    def forward(self, x):
        return self.backbone(x)  # [B, 2048, 16, 16]

model = ResNet50_Extractor().eval().to(device) # to(device) <-> cuda()

# 이미지 파일 목록
image_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

all_features = []

# 이미지별 feature 추출
for fname in tqdm(image_list):
    path = os.path.join(input_dir, fname)
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device) # to(device)<->cuda()  # [1, 3, 512, 512]

    with torch.no_grad():
        fmap = model(x)  # [1, 2048, 16, 16]
        patch_vectors = fmap.squeeze().reshape(2048, -1).T  # [256, 2048]
        all_features.append(patch_vectors)

# 모두 합쳐 저장
all_features_tensor = torch.cat(all_features, dim=0)  # [256 × num_images, 2048]
torch.save(all_features_tensor, os.path.join(output_dir, "all_patch_vectors_512.pt"))

print(f"✅ Saved patch features: {all_features_tensor.shape}")