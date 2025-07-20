import torch
import numpy as np
from scipy.interpolate import interp1d
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# ──────────────── 설정 ──────────────── #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(pretrained=True).to(device).eval()
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-2])

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ──────────────── 1. 기존 PDF → CDF 변환 ──────────────── #
pdf_raw = torch.load("./semantic_masks/semantic_mask_pdf.pt")
x_vals = pdf_raw["x"].numpy()
y_vals = pdf_raw["y"].numpy()

# 정규화된 PDF
pdf_norm = y_vals / np.sum(y_vals)
cdf = np.cumsum(pdf_norm)
cdf = cdf / cdf[-1]

# CDF 기반 penalty curve
penalty_curve = interp1d(x_vals, cdf, kind="linear", fill_value="extrapolate")

# ──────────────── 2. test 이미지 patch feature 추출 ──────────────── #
def extract_patch_feature(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        fmap = feature_extractor(x)  # [1, 2048, 16, 16]
        feat = fmap.flatten(2).squeeze(0).permute(1, 0)  # [256, 2048]
        feat = F.normalize(feat, dim=1)
    return feat

test_path = "dataset/test_test/test.png"
test_feat = extract_patch_feature(test_path)  # [256, 2048]

# ──────────────── 3. generic 벡터 불러오기 ──────────────── #
generic_feat = torch.load("dataset/generic_bank/all_patch_vectors_512.pt").to(device)
generic_feat = F.normalize(generic_feat, dim=1)

# ──────────────── 4. patch별 유사도 → penalty 매핑 ──────────────── #
sims = torch.matmul(test_feat, generic_feat.T)  # [256, K]
max_sim, _ = sims.max(dim=1)                    # [256]

sim_numpy = max_sim.cpu().numpy()
penalty_values = penalty_curve(sim_numpy)

penalty_tensor = torch.tensor(penalty_values, dtype=torch.float32)

# ──────────────── 5. 저장 ──────────────── #
save_path = "semantic_masks/test_patch_penalty_cdf.pt"
torch.save(penalty_tensor, save_path)
print(f"✅ Penalty 저장 완료: {save_path} ({penalty_tensor.shape})")
