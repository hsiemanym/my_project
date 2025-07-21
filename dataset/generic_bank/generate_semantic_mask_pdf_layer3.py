import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np

# 로드
sim_path = "./dataset/generic_bank/max_similarities_layer3.pt"
save_path = "./semantic_masks/semantic_mask_pdf_layer3.pt"
similarities = torch.load(sim_path).numpy()

# PDF 계산
pdf = gaussian_kde(similarities, bw_method=0.05)
x = np.linspace(0.3, 1.0, 500)
pdf_values = pdf(x)
pdf_values /= pdf_values.max()  # 정규화

# Penalty = 1 - PDF(similarity)
x_tensor = torch.tensor(x, dtype=torch.float32)
pdf_tensor = torch.tensor(1 - pdf_values, dtype=torch.float32)  # penalty로 사용

# 저장 (interpolation 가능하게)
torch.save({"x": x_tensor, "penalty": pdf_tensor}, save_path)
print(f"✅ Saved penalty map to {save_path}")

# 시각화
plt.plot(x, 1 - pdf_values)
plt.xlabel("Cosine Similarity")
plt.ylabel("Generic Penalty (1 - PDF)")
plt.title("Layer3 - PDF-based Generic Penalty")
plt.grid(True)
plt.savefig("./dataset/generic_bank/pdf_penalty_layer3.png")
plt.show()
