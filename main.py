import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from tqdm import tqdm

# 경로 설정
image_dir = "./dataset/Pretrain_Images/test_50"
output_path = "./dataset/generic_bank/all_patch_vectors_layer3.pt"

# ResNet50의 layer3까지 사용하는 모델 정의
class ResNetLayer3(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        self.out_channels = 1024

    def forward(self, x):
        return self.features(x)

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)

# 모델 준비
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetLayer3().to(device)
model.eval()

# 특징 벡터 추출
all_patch_features = []

with torch.no_grad():
    for fname in tqdm(os.listdir(image_dir)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        img_path = os.path.join(image_dir, fname)
        img = load_image(img_path).to(device)
        feature_map = model(img)  # [1, 1024, 32, 32]
        patches = feature_map.squeeze(0).permute(1, 2, 0).reshape(-1, model.out_channels)  # [1024, 1024]
        all_patch_features.append(patches.cpu())

# 텐서 저장
all_patch_tensor = torch.cat(all_patch_features, dim=0)  # [N*1024, 1024]
torch.save(all_patch_tensor, output_path)
print(f"✅ Saved: {all_patch_tensor.shape} to {output_path}")
