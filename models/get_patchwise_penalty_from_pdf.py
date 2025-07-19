import torch
import numpy as np
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from scipy.interpolate import interp1d
import os

# ---- 1. 설정 ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device).eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2])

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ---- 2. PDF 기반 penalty 곡선 로드 ----
pdf_curve_path = "./semantic_masks/semantic_mask_pdf.pt"
pdf_data = torch.load(pdf_curve_path)
x_vals = pdf_data["x"]  # similarity 값들 (예: np.linspace(0, 1, 100))
y_vals = pdf_data["y"]  # PDF 값들
penalty_curve = interp1d(x_vals, 1 - y_vals, kind='linear', fill_value="extrapolate")

# ---- 3. Generic Feature Bank 불러오기 ----
generic_path = "./dataset/generic_bank/all_patch_vectors_512.pt"
generic_vectors = torch.load(generic_path).to(device)  # [K, 2048]
generic_vectors = F.normalize(generic_vectors, dim=1)

# ---- 4. 테스트 이미지 patch-wise feature 추출 ----
def extract_patchwise_feature(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        feat_map = feature_extractor(x)  # [1, 2048, 16, 16]
        feat = feat_map.flatten(2).squeeze(0).permute(1, 0)  # [256, 2048]
        feat = F.normalize(feat, dim=1)
    return feat  # [256, 2048]

# ---- 5. 유사도 계산 및 penalty 변환 ----
test_img_path = "./dataset/test_test/test.png"
patch_features = extract_patchwise_feature(test_img_path)  # [256, 2048]
similarity_matrix = torch.matmul(patch_features, generic_vectors.T)  # [256, K]
top_similarities, _ = similarity_matrix.max(dim=1)  # [256]

# penalty 계산 (PDF 기반)
sim_numpy = top_similarities.cpu().numpy()  # [256]
penalties = penalty_curve(sim_numpy)        # [256]
penalty_tensor = torch.tensor(penalties, dtype=torch.float32)

# ---- 6. 저장 ----
save_path = "./semantic_masks/test_patch_penalty.pt"
torch.save(penalty_tensor, save_path)
print(f"✅ 저장 완료: {save_path} ({penalty_tensor.shape})")
