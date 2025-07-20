import os
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
import pickle

# ──────────────── 설정 ──────────────── #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 512
PATCH = 16  # resnet50 출력 기준 512/16 = 32 (flatten하면 1024 patches)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

resnet = models.resnet50(pretrained=True).to(device).eval()
backbone = torch.nn.Sequential(*list(resnet.children())[:-2])  # [B, 2048, H/32, W/32]

# ──────────────── 임베딩 추출 함수 ──────────────── #
def extract_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat_map = backbone(x)  # [1, 2048, 16, 16]
        patch_feats = feat_map.flatten(2).squeeze(0).permute(1, 0)  # [256, 2048]
        patch_feats = F.normalize(patch_feats, dim=1)
        global_feat = F.normalize(patch_feats.mean(dim=0, keepdim=True), dim=1)  # [1, 2048]
    return patch_feats, global_feat

# ──────────────── 1. Test 이미지 임베딩 ──────────────── #
test_path = "dataset/test_test/test.png"
test_patch, test_global = extract_embedding(test_path)

# ──────────────── 2. 전체 캐릭터 임베딩 비교 ──────────────── #
gallery_dir = "dataset/Pretrain_Images/test_50"
gallery_files = sorted([os.path.join(gallery_dir, f) for f in os.listdir(gallery_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])

gallery_feats = []
for path in tqdm(gallery_files, desc="Extracting Gallery Features"):
    _, global_feat = extract_embedding(path)
    gallery_feats.append(global_feat)

gallery_tensor = torch.cat(gallery_feats, dim=0)  # [N, 2048]

# ──────────────── 3. Top-1 Retrieval ──────────────── #
sims = torch.matmul(test_global, gallery_tensor.T).squeeze(0)  # [N]
top1_idx = torch.argmax(sims).item()
top1_path = gallery_files[top1_idx]
print(f"✅ Top-1 유사한 이미지: {top1_path} (Similarity: {sims[top1_idx]:.4f})")

# ──────────────── 4. Top-1 이미지 patch-wise 임베딩 ──────────────── #
top1_patch, _ = extract_embedding(top1_path)

# ──────────────── 5. 저장 ──────────────── #
out_dir = "dataset/test_test"
os.makedirs(out_dir, exist_ok=True)
torch.save(test_patch, os.path.join(out_dir, "test_patch.pt"))
torch.save(top1_patch, os.path.join(out_dir, "top1_patch.pt"))
torch.save(top1_patch, os.path.join(out_dir, "top1_path.pt"))

print("✅ Patch-wise feature 저장 완료:")
print("- test_patch.pt")
print("- top1_patch.pt")
print("- top1_path.pt")
