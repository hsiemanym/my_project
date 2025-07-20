import torch
import torch.nn.functional as F

# ──────────────── 경로 설정 ──────────────── #
test_patch_path = "dataset/test_test/test_patch.pt"
top1_patch_path = "dataset/test_test/top1_patch.pt"
penalty_path = "semantic_masks/test_patch_penalty_cdf.pt"  # ✅ 새로 만든 CDF 기반 penalty

# ──────────────── 1. 파일 불러오기 ──────────────── #
test_feat = torch.load(test_patch_path)      # [256, 2048]
top1_feat = torch.load(top1_patch_path)      # [256, 2048]
penalty = torch.load(penalty_path)           # [256], 값은 0 ~ 1 사이

# normalize
test_feat = F.normalize(test_feat, dim=1)
top1_feat = F.normalize(top1_feat, dim=1)

# ──────────────── 2. patch-wise similarity 계산 ──────────────── #
sim = F.cosine_similarity(test_feat, top1_feat, dim=1)  # [256]

# ──────────────── 3. penalty weight 적용 ──────────────── #
weight = 1 - penalty  # generic할수록 가중치 줄임
weighted_sim = sim * weight

# ──────────────── 4. 최종 score 계산 ──────────────── #
final_score = weighted_sim.mean().item()
print(f"\n✅ Generic-aware Similarity Score: {final_score:.4f}")
