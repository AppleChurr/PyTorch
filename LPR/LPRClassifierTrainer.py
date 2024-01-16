import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset, Dataset

import onnx
import onnxruntime
from onnx2pytorch import ConvertModel

from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm  # 추가
import os

import mManager as mManager

from torchvision.models import resnet18, ResNet18_Weights


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        print("Data Path : " + self.root_dir)

        # 모든 하위 폴더를 순회
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                # 폴더 이름에서 클래스 레이블 추출
                label = int(folder_name.split('_')[0])
                self.load_data(folder_path, label)

    def load_data(self, folder, label):
        for filename in os.listdir(folder):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(folder, filename)
                self.file_paths.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class DataLoaderWrapper:
    def __init__(self, dataset_cls, root_dir, transform, batch_size=16, test_size=0.4, random_state=42):
        self.dataset = dataset_cls(root_dir, transform)
        self.batch_size = batch_size
        self.test_size = test_size
        self.random_state = random_state
        self._prepare_data_loaders()

    def _prepare_data_loaders(self):
        # 레이블 배열 준비
        labels = [label for _, label in self.dataset]

        # 데이터 분할
        train_idx, val_test_idx = train_test_split(range(len(self.dataset)), test_size=self.test_size, stratify=labels, random_state=self.random_state)
        val_idx, test_idx = train_test_split(val_test_idx, test_size=0.5, stratify=[labels[i] for i in val_test_idx], random_state=self.random_state)

        # Subset을 사용하여 데이터셋 분할
        train_dataset = Subset(self.dataset, train_idx)
        val_dataset = Subset(self.dataset, val_idx)
        test_dataset = Subset(self.dataset, test_idx)

        # 데이터로더 설정
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def get_loaders(self):
        return self.train_loader, self.val_loader, self.test_loader

# 모델 평가 함수
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    average_loss = total_loss / len(dataloader)
    return average_loss, accuracy

# 데이터 전처리 및 Augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # 데이터 증강
    transforms.RandomRotation(10),      # 데이터 증강
    transforms.ToTensor(),
])

# 데이터 폴더 경로 설정
TrainData_root = "D:\\20_Data\\50_번호판 데이터\\[2021] 화성 단일\\"

# 데이터 로더 클래스 인스턴스화
data_loader_wrapper = DataLoaderWrapper(CustomDataset, TrainData_root, transform, batch_size=16)

# 데이터 로더 가져오기
train_loader, val_loader, test_loader = data_loader_wrapper.get_loaders()


# ONNX 모델 경로 및 기본 모델 설정
onnx_model_path = "./best_model_step0.onnx"

# model = load_pretrained_model(onnx_model_path)
_modelManager = mManager.Model_Step0(onnx_model_path)
_modelManager.Init_model()

model = _modelManager.get_model()

# 모델을 훈련 모드로 설정
model.train()

if torch.cuda.is_available:
    print("Cuda is available")
else:
    print("Cuda is not available")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 최적의 모델 저장을 위한 초기 설정
best_loss = float('inf')

num_epochs = 10

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 검증 단계
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 검증 손실 및 정확도 계산
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.2f}%")

    # 테스트 손실 및 정확도 계산
    test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
    print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.2f}%")

    # 최적의 모델 저장
    if test_loss < best_loss:
        best_loss = test_loss
        save_model(model, onnx_model_path)
        print(f"ONNX Model saved to {onnx_model_path}")