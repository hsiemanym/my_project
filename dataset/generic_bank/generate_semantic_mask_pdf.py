import torch
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import os

# cosine similarity 벡터 불러오기
similarity_tensor = torch.load("./dataset/generic_bank/max_similarities.pt")  # shape: [N]
similarities = similarity_tensor.cpu().numpy()

# KDE 기반 PDF 추정
kde = gaussian_kde(similarities)

# 정밀한 x 축 생성 (0~1 구간을 100개로 나눔)
x_vals = np.linspace(0, 1, 100)
pdf_curve = kde(x_vals)

# 저장: dict 형태로 (나중에 interpolation에 활용)
os.makedirs("semantic_masks", exist_ok=True)
torch.save({
    "x": torch.tensor(x_vals, dtype=torch.float32),
    "y": torch.tensor(pdf_curve, dtype=torch.float32)
}, "semantic_masks/semantic_mask_pdf.pt")

# 시각화
plt.figure(figsize=(8, 4))
plt.scatter(similarities, 1 - kde(similarities), s=5, alpha=0.4)
plt.xlabel("Cosine Similarity")
plt.ylabel("Generic Penalty (1 - PDF)")
plt.title("PDF-based Genericness Penalty")
plt.grid(True)
plt.tight_layout()
plt.show()
