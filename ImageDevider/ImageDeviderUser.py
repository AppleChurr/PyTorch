import onnx
import onnxruntime
from torchvision import transforms
from PIL import Image
import os
import shutil
from tqdm import tqdm

# ONNX 모델 로드 함수
def load_onnx_model(onnx_path):
    return onnxruntime.InferenceSession(onnx_path)

# 이미지 분류 함수
def classify_image(onnx_model, image_path, transform):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).numpy()

    # ONNX 모델의 입력 노드 이름 확인
    input_name = onnx_model.get_inputs()[0].name

    # ONNX 모델을 사용하여 추론
    output = onnx_model.run(None, {input_name: image.astype('float32')})
    predicted = int(output[0][0].argmax())
    return predicted

# 이미지 및 txt 파일 이동 함수
def move_images_and_txt(data_folder, onnx_model_path):
    # ONNX 모델 로드
    onnx_model = load_onnx_model(onnx_model_path)

    # 모델 및 전처리 transform 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 폴더 경로 설정
    true_folder = os.path.join(data_folder, "True")
    false_folder = os.path.join(data_folder, "False")

    # 하위 폴더 생성
    os.makedirs(true_folder, exist_ok=True)
    os.makedirs(false_folder, exist_ok=True)

    # 모든 폴더 검색 (재귀적으로)
    for root, dirs, files in os.walk(data_folder):
        
        # "True" 또는 "False"라는 폴더를 가진 경우 검색에서 제외
        if "True" in dirs or "False" in dirs:
            continue

        # 현재 폴더 내의 이미지 및 txt 파일 리스트 가져오기
        image_files = [f for f in files if f.endswith(('.jpg', '.jpeg', '.png'))]

        # Use tqdm for progress tracking
        with tqdm(total=len(image_files), desc=f'Classifying and moving images in {root}') as pbar:
            # 이미지 및 txt 파일을 True 및 False 폴더로 이동
            for image_file in image_files:
                txt_file = os.path.splitext(image_file)[0] + '.txt'

                if txt_file in files:
                # 이미지 분류
                    image_path = os.path.join(root, image_file)
                    prediction = classify_image(onnx_model, image_path, transform)

                    # 이미지 파일 이동
                    src_image_path = os.path.join(root, image_file)
                    dest_image_path = os.path.join(true_folder if prediction == 1 else false_folder, image_file)
                    shutil.move(src_image_path, dest_image_path)

                    # txt 파일 이동
                    src_txt_path = os.path.join(root, txt_file)
                    dest_txt_path = os.path.join(true_folder if prediction == 1 else false_folder, txt_file)
                    shutil.move(src_txt_path, dest_txt_path)

                pbar.update(1)  # Update progress bar

    print("Images and txt files moved based on ONNX model predictions.")



# Data 폴더 경로 설정
data_folder_path = "Data"  # 실제 경로로 변경
# ONNX 모델 경로 설정
onnx_model_path = "best_model_epoch_4_loss_0.0725.onnx"  # 모델 경로는 저장한 모델에 맞게 설정

# 이미지 및 txt 파일 이동 함수 호출
move_images_and_txt(data_folder_path, onnx_model_path)
