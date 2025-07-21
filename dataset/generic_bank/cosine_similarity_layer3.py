import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# 경로
feature_path = "./dataset/generic_bank/all_patch_vectors_layer3.pt"
save_path = "./dataset/generic_bank/max_similarities_layer3.pt"

# 데이터 로드
features = torch.load(feature_path)  # [N, D]
features = F.normalize(features, dim=1)

# 코사인 유사도 계산
similarities = torch.matmul(features, features.T)  # [N, N]
mask = torch.eye(similarities.size(0), device=similarities.device).bool()
similarities.masked_fill_(mask, -1.0)  # 자기 자신은 제외

# 가장 높은 유사도 선택
max_similarities = similarities.max(dim=1).values.cpu()  # [N]
torch.save(max_similarities, save_path)
print(f"✅ Saved max similarities to {save_path}")

# 히스토그램 시각화
plt.hist(max_similarities.numpy(), bins=80)
plt.xlabel("Cosine similarity")
plt.ylabel("Number of patches")
plt.title("Layer3 - Cosine Similarity to Closest Generic Feature")
plt.tight_layout()
plt.savefig("./dataset/generic_bank/hist_similarity_layer3.png")
plt.show()
