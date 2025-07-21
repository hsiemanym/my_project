import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np


# 경로 설정
img_paths = {
    "test": "./dataset/test_test/test.png",
    "top1": "./dataset/Pretrain_Images/test_50/005.jpg"
}
penalty_save_dir = "./semantic_masks"
pdf_path = "./semantic_masks/semantic_mask_pdf_layer3.pt"

# 1. PDF 기반 Penalty 로드
pdf_data = torch.load(pdf_path)
x_grid = pdf_data["x"]          # [500]
penalty_grid = pdf_data["penalty"]  # [500]

# 2. 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 3. Feature 추출기 (ResNet50의 layer3)
resnet = models.resnet50(pretrained=True)
layer_outputs = {}

def hook_fn(module, input, output):
    layer_outputs['features'] = output

resnet.layer3.register_forward_hook(hook_fn)
resnet.eval()

# 4. Cosine similarity 기반 penalty 계산 함수
def compute_penalty(img_tensor):
    with torch.no_grad():
        _ = resnet(img_tensor.unsqueeze(0))  # forward
        features = layer_outputs['features']  # [1, C, H, W]

    features = F.adaptive_avg_pool2d(features, (32, 32))  # 보존된 해상도
    B, C, H, W = features.shape
    patches = features[0].permute(1, 2, 0).reshape(H*W, C)  # [1024, C]
    patches = F.normalize(patches, dim=1)

    # patch-wise 코사인 유사도 최대값
    similarity = torch.matmul(patches, patches.T)
    similarity.fill_diagonal_(-1.0)
    max_sim = similarity.max(dim=1).values  # [1024], torch.Tensor

    # numpy로 변환 후 선형 보간
    penalty_np = np.interp(max_sim.cpu().numpy(), x_grid.numpy(), penalty_grid.numpy())
    return torch.tensor(penalty_np, dtype=torch.float32)

# 5. 이미지별 penalty 계산 및 저장
for name, path in img_paths.items():
    img = Image.open(path).convert("RGB")
    img_tensor = transform(img)

    penalty = compute_penalty(img_tensor)
    torch.save(penalty, os.path.join(penalty_save_dir, f"{name}_patch_penalty_layer3.pt"))
    print(f"✅ Saved {name} penalty: {penalty.shape}")
