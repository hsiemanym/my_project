# generic feature 저장 및 로드


# GPU 설정
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

#1 KMeans로 generic cluster 중심 구하기
# patch feature 불러오기
features = torch.load("./dataset/generic_bank/all_patch_vectors_512.pt")  # [12800, 2048]
features_np = features.cpu().numpy()

# KMeans 클러스터링
k = 100  # generic feature vector 개수
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
kmeans.fit(features_np)
generic_vectors = torch.tensor(kmeans.cluster_centers_)  # [100, 2048]


#2 cosine similarity 계산
# 정규화
f_norm = F.normalize(features, dim=1)  # [12800, 2048]
g_norm = F.normalize(generic_vectors, dim=1)  # [100, 2048]

# 모든 patch와 모든 generic vector 간 cosine similarity
similarity_matrix = torch.matmul(f_norm, g_norm.T)  # [12800, 100]

# patch당 가장 유사한 generic feature 하나만 고려
max_similarities, _ = similarity_matrix.max(dim=1)  # [12800]



#3 히스토그램 시각화 & threshold 선택
# 시각화
plt.hist(max_similarities.cpu().numpy(), bins=100)
plt.title("Cosine Similarity to Closest Generic Feature")
plt.xlabel("Cosine similarity")
plt.ylabel("Number of patches")
plt.grid(True)
plt.show()


#4 Threshold 저장
threshold_value = 0.90  # 예시
with open("cosine_threshold.json", "w") as f:
    json.dump({"cosine_threshold": threshold_value}, f)
    

torch.save(max_similarities, "./dataset/generic_bank/max_similarities.pt")
