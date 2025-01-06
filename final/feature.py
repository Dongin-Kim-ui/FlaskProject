import os
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ResNet 모델 불러오기 (사전 훈련된 가중치 사용)
resnet = models.resnet50(pretrained=True)
# 마지막 레이어를 제거하여 특징 벡터를 추출할 준비를 합니다.
resnet_feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
# 모델을 평가 모드로 설정
resnet_feature_extractor.eval()

# 이미지를 불러오고 전처리하는 함수
def preprocess_image(image_path):
    # 이미지를 RGB 모드로 열기
    img = Image.open(image_path).convert('RGB')
    # 이미지를 정해진 크기(224x224)로 resize하고 텐서로 변환
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = preprocess(img)
    # 배치 차원 추가
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# 이미지 폴더 경로
image_folder = 'c:/Users/user/Desktop/여행 이미지/새 폴더'

# 이미지 파일 리스트 가져오기
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 특징 벡터를 저장할 리스트
feature_vectors = []

# 이미지 파일들에 대해 반복하며 특징 벡터 추출
for image_file in image_files:
    # 이미지 전처리
    input_image = preprocess_image(image_file)
    
    # 특징 벡터 추출
    with torch.no_grad():
        output_features = resnet_feature_extractor(input_image)
    
    # 특징 벡터를 numpy 배열로 변환하여 리스트에 추가
    feature_vectors.append(output_features.squeeze().numpy())

# 특징 벡터 리스트를 numpy 배열로 변환
feature_vectors = np.array(feature_vectors)

# npz 파일로 저장
np.savez('feature_vectors1.npz', features=feature_vectors)
