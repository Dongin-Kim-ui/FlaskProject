from flask import Flask, render_template, request, redirect, url_for, jsonify,Response
import boto3
import os
from ultralytics import YOLO
import tempfile
from PIL import Image
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import xml.etree.ElementTree as ET
from flask import redirect
import yaml
from multiprocessing import freeze_support
import glob
import shutil
import ultralytics
import urllib.parse
import time
from botocore.exceptions import ClientError

app = Flask(__name__)

# AWS S3 related settings
S3_BUCKET = 'completesite'
S3_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
S3_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

s3 = boto3.client(
    's3',
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY
)

# Temporary directory path
TMP_DIR = tempfile.gettempdir()

tmp_dir = tempfile.mkdtemp()


# 특징을 추출하는 함수
def extract_features(image_path):
    try:
        # 사전 학습된 ResNet 모델 불러오기
        resnet = models.resnet50(pretrained=True)
        resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
        # 평가 모드로 설정 (추론 모드)
        resnet.eval()

        # 이미지 불러오기 및 전처리
        image = Image.open(image_path)
        preprocess = transforms.Compose([
            transforms.Resize(256),  # 이미지를 256 크기로 조정
            transforms.CenterCrop(224),  # 중앙을 224 크기로 자르기
            transforms.ToTensor(),  # 이미지를 텐서로 변환
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 정규화
        ])
        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가

        # 특징 추출
        with torch.no_grad():  # 역전파 비활성화 (메모리 절약 및 속도 향상)
            features = resnet(image_tensor)
        features = features.squeeze().numpy()  # 불필요한 차원 제거 및 numpy 배열로 변환

        return features  # 특징 벡터 반환
    except Exception as e:
        print(e)  # 예외 발생 시 오류 메시지 출력
        return None  # 예외 발생 시 None 반환

    
def get_first_image():
    # 지정된 경로에서 JPEG 이미지 검색
    image_paths = glob.glob("runs/segment/predict/*.jpg")

    # 이미지가 있는지 확인
    if not image_paths:
        return None  # 또는 적절하게 처리 (예: 상황 로그 기록 또는 사용자에게 알림)

    # 리스트에서 첫 번째 이미지 반환
    return image_paths[0]

def get_latest_yolo_image_key(prefix):
    # S3 버킷에서 주어진 접두사(prefix)를 가진 객체 목록을 가져옴
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    try:
        # 객체 목록 중에서 가장 최근에 수정된 객체를 찾음
        latest_object = max(response.get('Contents', []), key=lambda x: x['LastModified'])
        return latest_object['Key']  # 가장 최근 객체의 키를 반환
    except ValueError:
        return None  # 객체 목록이 비어 있을 경우 None을 반환


# 최신 업로드된 이미지의 키(경로)를 가져오는 함수
def get_latest_uploaded_image_key():
    try:
        # S3 버킷에서 'uploaded/' 접두사를 가진 객체 목록을 가져옴
        response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='uploaded/')
        # 객체 목록 중에서 가장 최근에 수정된 객체를 찾음
        latest_object = max(response['Contents'], key=lambda x: x['LastModified'])
        latest_key = latest_object['Key']  # 가장 최근 객체의 키를 저장
        return latest_key  # 가장 최근 객체의 키를 반환
    except Exception as e:
        print(e)  # 예외 발생 시 오류 메시지 출력
        return None  # 예외 발생 시 None 반환

    
def remove_duplicates(words, confidences):
    # words와 confidences 리스트의 길이가 같은지 확인
    if len(words) != len(confidences):
        raise ValueError("Words and confidences must have the same length")
    
    # 단어와 해당 confidence 값을 저장할 딕셔너리
    word_conf_map = {}
    for i, word in enumerate(words):
        if word in word_conf_map:
            # 기존 단어와 비교하여 더 높은 confidence 값을 가진 단어를 유지
            if confidences[i] > word_conf_map[word][1]:
                word_conf_map[word] = (i, confidences[i])
        else:
            # 새로운 단어와 해당 인덱스 및 confidence 값을 추가
            word_conf_map[word] = (i, confidences[i])
    
    # 유지할 인덱스 집합 생성
    indices_to_keep = set(index for index, _ in word_conf_map.values())
    
    # 인덱스를 기반으로 단어와 confidence 값을 필터링
    filtered_words = [words[i] for i in range(len(words)) if i in indices_to_keep]
    filtered_confidences = [confidences[i] for i in range(len(words)) if i in indices_to_keep]
    
    return filtered_words, filtered_confidences
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/image')
def index():
    return render_template('recommend_similar_images.html')

@app.route('/yolo_home')
def yolo_home():
    return render_template('YOLO.html')

@app.route('/upload', methods=['POST'])
def upload():
    # 요청 파일에 'file'이 없으면 인덱스 페이지로 리다이렉트
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    # 파일명이 비어있으면 인덱스 페이지로 리다이렉트
    if file.filename == '':
        return redirect(url_for('index'))
    
    try:
        # S3에 파일 업로드
        s3.upload_fileobj(file, S3_BUCKET, 'uploaded/' + file.filename)
        
        # 최근 업로드된 이미지 키 가져오기
        latest_uploaded_image_key = get_latest_uploaded_image_key()
        if latest_uploaded_image_key is None:
            return "Failed to get the latest uploaded image."

        # S3에서 최근 업로드된 이미지 가져오기
        image_bytes = s3.get_object(Bucket=S3_BUCKET, Key=latest_uploaded_image_key)['Body'].read()
        image = Image.open(BytesIO(image_bytes))
        
        # 이미지를 임시 디렉토리에 저장
        image_path = os.path.join(TMP_DIR, 'uploaded_image.jpg')
        image.save(image_path)

        # 이미지에서 특징 추출
        features = extract_features(image_path)
        if features is None:
            return "Failed to extract features from the uploaded image."

        # 특징을 임시 파일에 저장
        features_file_path = os.path.join(TMP_DIR, 'features.npz')
        np.savez(features_file_path, features=features)

        # 특징 파일을 S3에 업로드
        features_key = 'features/' + latest_uploaded_image_key.split('/')[-1].split('.')[0] + '.npz'
        s3.upload_file(features_file_path, S3_BUCKET, features_key)

        # 임시 파일 삭제
        os.remove(image_path)
        os.remove(features_file_path)

        # 최근 업로드된 이미지 키와 관련된 특징 키 가져오기
        latest_uploaded_image_key = get_latest_uploaded_image_key()
        latest_features_key = 'features/' + latest_uploaded_image_key.split('/')[-1].split('.')[0] + '.npz'
        latest_features_obj = s3.get_object(Bucket=S3_BUCKET, Key=latest_features_key)
        latest_features = np.load(BytesIO(latest_features_obj['Body'].read()))['features']

        similar_folder_features = []
        
        # S3에서 특정 폴더의 모든 .npz 파일 가져오기
        folder_objects = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix='npz_folder/')
        for obj in folder_objects['Contents']:
            if obj['Key'].endswith('.npz'):
                obj_body = s3.get_object(Bucket=S3_BUCKET, Key=obj['Key'])['Body']
                folder_features = np.load(BytesIO(obj_body.read()))['features']

                # 특징 벡터의 크기 조정
                max_shape = max(latest_features.shape, folder_features.shape)
                resized_latest_features = np.zeros(max_shape)
                resized_folder_features = np.zeros(max_shape)
                resized_latest_features[:latest_features.shape[0]] = latest_features
                resized_folder_features[:folder_features.shape[0]] = folder_features

                similar_folder_features.append(resized_folder_features)

        similar_folder_features = np.array(similar_folder_features)
        if similar_folder_features.size == 0:
            return "No valid feature vectors found for comparison."

        # 코사인 유사도로 유사도 점수 계산
        similarity_scores = cosine_similarity(resized_latest_features.reshape(1, -1), similar_folder_features)
        top_indices = similarity_scores.argsort()[0][-3:][::-1]

        similar_images = []
        while len(similar_images) < 3:
            for index in top_indices:
                similar_image_key = folder_objects['Contents'][index]['Key']
                similar_image_key = similar_image_key.replace('.npz', '.jpg')
                similar_image_url = s3.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': similar_image_key}, ExpiresIn=3600)
                similar_images.append(similar_image_url)
                if len(similar_images) >= 3:
                    break

        return render_template('recommend_similar_images.html', result_image_1=similar_images[0], result_image_2=similar_images[1], result_image_3=similar_images[2])
    
    # 예외 처리
    except Exception as e:
        print("Error:", e)
        return "Failed to recommend similar images."

    
# 이미지 업로드 및 처리 함수
def upload_image():
    image_file = request.files['file']  # 요청에서 'file' 필드의 파일을 가져옴
    # 이미지 처리 시뮬레이션
    for i in range(5):
        # 진행 상황을 전역 변수나 적절한 메커니즘으로 저장 (예제는 단순화)
        progress = int((i + 1) / 5 * 100)
        time.sleep(1)  # 처리 시간 시뮬레이션
    return jsonify({'message': '이미지가 성공적으로 처리되었습니다.'})  # 처리 완료 후 메시지 반환

# 진행 상황을 전송하는 엔드포인트
@app.route('/progress')
def progress():
    def generate():
        for i in range(5):
            yield f"data: {i * 20}\n\n"  # 진행 상황을 데이터 스트림 형식으로 전송
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')  # 이벤트 스트림으로 응답


@app.route('/yolo', methods=['POST'])
def yolo():
    try:
        # 요청 파일에서 'file'을 가져옴
        if 'file' in request.files:
            file = request.files.get('file')
        if not file or not file.filename:
            return redirect(url_for('index'))
    
        # 파일을 임시 디렉토리에 저장
        file_path = os.path.join(tmp_dir, file.filename)
        file.save(file_path)

        # S3에 파일 업로드
        s3.upload_file(file_path, S3_BUCKET, 'yolo_upload/' + file.filename)
        latest_yolo_image_key = get_latest_yolo_image_key('yolo_upload/')
        if not latest_yolo_image_key:
            return "Failed to get the latest yolo_upload image."

        # S3에서 이미지 다운로드
        image_bytes = s3.get_object(Bucket=S3_BUCKET, Key=latest_yolo_image_key)['Body'].read()
        image = Image.open(BytesIO(image_bytes))

        # 이미지를 임시 파일로 저장
        image_path = os.path.join(tmp_dir, file.filename)
        image.save(image_path)
        print('end before load model')

        # YOLO 예측 수행
        model = YOLO('best.pt')
        results = model.predict(source=image_path, save=True)

        global cls_list
        cls_list = []
        conf_list = []

        names_yolo = {}
        for result in results:
            # result가 Results 객체인지 확인
            if hasattr(result, 'boxes'):
                # boxes 속성에 안전하게 접근
                boxes = result.boxes
                for box in boxes:
                    if box.cls.numel() == 1:
                        cls_list.append(box.cls.item())
                    if box.conf.numel() == 1:
                        conf_list.append(box.conf.item())

            # names 속성이 있는지 확인
            if 'names' in dir(result):
                names_yolo.update(result.names)

        global names_yolo_list
        names_yolo_list = []

        # cls_list에 있는 클래스 ID를 names_yolo 딕셔너리에서 이름으로 변환
        for cls_id in cls_list:
            if cls_id in names_yolo:
                names_yolo_list.append(names_yolo[cls_id])
            else:
                names_yolo_list.append(None)

        print(names_yolo_list)

        # 중복 제거 및 필터링
        filtered_words, filtered_confidences = remove_duplicates(names_yolo_list, conf_list)

        if filtered_confidences:
            top_indices = sorted(range(len(filtered_confidences)), key=lambda i: filtered_confidences[i], reverse=True)[:3]
            cls_list = [filtered_words[i] for i in top_indices]
            conf_list = [filtered_confidences[i] for i in top_indices]
        else:
            cls_list = []
            conf_list = []

        # cls_list와 conf_list의 길이를 3으로 맞춤
        while len(cls_list) < 3:
            cls_list.append(None)
        while len(conf_list) < 3:
            conf_list.append(None)
    
        global f_conf_list
        f_conf_list = [round(conf, 4) if conf is not None else None for conf in conf_list]

        # 결과 이미지 가져오기
        result_image = get_first_image()
        filename = os.path.basename(result_image)

        # 결과 이미지를 S3에 업로드
        with open(result_image, 'rb') as file:
            s3.upload_fileobj(file, S3_BUCKET, f'yolo_result/{filename}')
            print("Image uploaded to S3.")

        # 로컬 결과 이미지 디렉토리 제거
        if os.path.exists("runs"):
            shutil.rmtree("runs")
            print("Local result directory removed.")

        # S3에서 최근 YOLO 결과 이미지 키 가져오기
        key = get_latest_yolo_image_key('yolo_result/')

        print(cls_list)
        print(f_conf_list)

        # S3에서 결과 이미지 데이터 가져오기
        img_url = s3.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET, 'Key': key}, ExpiresIn=3600)
        print("Image data retrieved from S3.")

        # 결과를 템플릿에 전달
        return render_template('YOLO.html', result_image_1=img_url)
    
    except Exception as e:
        print("Error:", e)
        return "Failed to yolo images."

    
@app.route('/gauge-values')
def gauge_values():
    data = {
        "class_name_1": cls_list[0],
        "value_1": f_conf_list[0],
        "class_name_2":  cls_list[1],
        "value_2": f_conf_list[1],
        "class_name_3":  cls_list[2], 
        "value_3": f_conf_list[2]
    }
    # 값이 None이 아닌 경우에만 * 100을 수행
    for key in ["value_1", "value_2", "value_3"]:
        if data[key] is not None:
            data[key] *= 100.0

    return jsonify(data)



@app.route('/metadata/<image_name>')
def metadata(image_name):
    try:
        # S3에서 LandmarkDB.xml 파일 불러오기
        metadata_key = 'metadata/LandmarkDB.xml'
        metadata_file = s3.get_object(Bucket=S3_BUCKET, Key=metadata_key)
        metadata_content = metadata_file['Body'].read()

        # XML 메타데이터 파싱
        root = ET.fromstring(metadata_content)

        # 클릭된 이미지에 대한 메타데이터 찾기
        for landmark in root.findall('LandmarkDB'):
            imagefile_name = landmark.find('imagefile_name').text
            if imagefile_name == image_name:
                latitude = landmark.find('latitude').text
                longitude = landmark.find('longitude').text

                # 위도와 경도를 포함한 Google Maps URL 생성
                map_url = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

                # 사용자를 새로운 창에서 Google Maps URL로 리다이렉트
                return redirect(map_url, code=302)

        # 클릭된 이미지에 대한 메타데이터를 찾지 못한 경우
        return "Metadata not found for the clicked image."

    # S3 클라이언트 오류 처리
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            return jsonify({"error": "Metadata file not found."}), 404
        else:
            return jsonify({"error": str(e)}), 500
    # 기타 예외 처리
    except Exception as e:
        return jsonify({"error": "An unexpected error occurred: " + str(e)}), 500

    



# 이하 함수는 S3 또는 다른 데이터 소스에서 메타데이터를 가져오는 코드입니다.
def find_metadata(image_name):
    try:
        # 이미지 이름이 누락되었거나 비어 있는지 확인합니다.
        if not image_name:
            return {"error": "Image name is missing."}

        # 메타데이터 파일을 불러옵니다.
        metadata_key = 'metadata/LandmarkDB.xml'
        metadata_file = s3.get_object(Bucket=S3_BUCKET, Key=metadata_key)
        metadata_content = metadata_file['Body'].read()

        # XML 형식의 메타데이터를 파싱합니다.
        root = ET.fromstring(metadata_content)

        # 이미지 이름에 해당하는 메타데이터를 찾습니다.
        for landmark in root.findall('LandmarkDB'):
            if landmark.find('imagefile_name').text == image_name:
                landmark_ex = landmark.find('landmark_ex').text
                image_name_without_extension = os.path.splitext(image_name)[0]
                return {"imagefile_name": image_name_without_extension, "landmark_ex": landmark_ex}        
        
        # 메타데이터를 찾지 못한 경우
        return {"error": "Metadata not found for the image: " + image_name}
    except Exception as e:
        return {"error": str(e)}

@app.route('/get_image_metadata_by_name', methods=['POST'])
def get_image_metadata_by_name():
    try:
        # 클라이언트로부터 이미지 이름을 받습니다.
        data = request.json
        image_name_encoded = data.get('image_name')

        # 이미지 이름이 유효한지 확인하고 URL 디코딩합니다.
        if not image_name_encoded or not isinstance(image_name_encoded, str):
            return jsonify({"error": "Invalid image name."}), 400
        image_name = urllib.parse.unquote(image_name_encoded)

        # 이미지 이름에서 서명 정보를 제외합니다.
        image_name = image_name.split("?")[0]

        # 이미지 이름을 사용하여 메타데이터를 찾습니다.
        metadata = find_metadata(image_name)

        # 메타데이터를 JSON 형식으로 반환합니다.
        return jsonify(metadata)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    freeze_support()
    app.run(debug=True, port= 5000)
