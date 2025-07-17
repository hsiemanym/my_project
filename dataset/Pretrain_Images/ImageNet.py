

# GPU 설정
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import os
from PIL import Image
import torch
from tqdm import tqdm
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# 1. CLIP 모델 로딩
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. CLIP 텍스트 쿼리 정의
text_descriptions = ["a rabbit in A-pose", "a rabbit crouching"]
text_tokens = clip.tokenize(text_descriptions).to(device)

# 3. 경로 설정 (📌 여기만 맞게 수정)
input_dir = "/home/hoeyeon/my_project/dataset/Pretrain_Images/ImageNet/rabbit"
output_dir = "/home/hoeyeon/my_project/dataset/Pretrain_Images/ImageNet/rabbit_Apose"
os.makedirs(output_dir, exist_ok=True)

# 4. 이미지 필터링
for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    try:
        image_path = os.path.join(input_dir, fname)
        image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text_tokens)

            similarity = (image_features @ text_features.T).softmax(dim=-1).squeeze().cpu().numpy()

        # A-pose 유사도가 가장 높고, 0.5 이상일 때만 저장
        if similarity.argmax() == 0 and similarity[0] > 0.35:
            Image.open(image_path).save(os.path.join(output_dir, fname))

    except Exception as e:
        continue
