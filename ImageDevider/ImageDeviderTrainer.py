import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import multiprocessing
import torch.onnx
from tqdm import tqdm  # 추가
from sklearn.model_selection import train_test_split

# 사용자 정의 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_paths = []
        self.labels = []

        print("Data Path : " + self.root_dir)

        # "True" 폴더에 대한 데이터 로딩
        true_folder = os.path.join(root_dir, "True")
        true_label = 1  # "True"를 나타내는 클래스 레이블
        self.load_data(true_folder, true_label)

        # "False" 폴더에 대한 데이터 로딩
        false_folder = os.path.join(root_dir, "False")
        false_label = 0  # "False"를 나타내는 클래스 레이블
        self.load_data(false_folder, false_label)

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


if __name__ == '__main__':
    # 데이터 폴더 경로 설정
    TrainData_root = "D:\\02_PyTorch\\ImageDevider\\TrainData\\"

    # 데이터 전처리 및 Augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 전체 데이터를 Train과 Validation으로 나누기
    full_dataset = CustomDataset(TrainData_root, transform)

    # 전체 데이터의 일부를 Validation 데이터로 사용
    train_dataset, val_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)

    # 남은 데이터를 다시 Train과 Test로 나누기
    train_dataset, test_dataset = train_test_split(train_dataset, test_size=0.25, random_state=42)

    # 동일한 batch_size 사용
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


    # CNN 모델 정의
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(32 * 56 * 56, 512)
            self.fc2 = nn.Linear(512, 2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 32 * 56 * 56) 
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 생성 및 손실 함수, 최적화 함수 정의
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 함수로 분리된 평가 및 정확도 계산 부분
    def evaluate_model(model, dataloader, device):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    # 초기 최소 손실을 양의 무한대로 설정
    best_loss = float('inf')

    # 학습
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        with tqdm(data_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as t:  # tqdm 추가
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                t.set_postfix(loss=total_loss / (t.n + 1))  # tqdm 업데이트
                    
        # 테스트 손실 계산
        test_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

        # 현재 에폭에서의 테스트 손실 출력
        test_loss /= len(test_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}")

        # 현재 에폭에서의 테스트 정확도 출력
        test_accuracy = evaluate_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {test_accuracy * 100:.2f}%")


        # 현재 테스트 손실이 이전 최소 손실보다 낮으면 모델을 저장
        if test_loss < best_loss:
            best_loss = test_loss
            # ONNX로 모델 내보내기
            onnx_path = f"best_model_epoch_{epoch + 1}_loss_{test_loss:.4f}.onnx"
            example_input = torch.randn(1, 3, 224, 224).to(device)
            torch.onnx.export(model, example_input, onnx_path, verbose=True)
            print(f"ONNX Model saved to {onnx_path}")