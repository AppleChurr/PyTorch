import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
import os

class ModelManager:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def create_model(self, *args, **kwargs):
        # 모델 생성 로직
        pass

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path))

    def Init_model(self, *args, **kwargs):
        # 사전 훈련된 모델 로드 로직
        pass

    def get_model(self):
        return self.model
    
    def print_model(self):
        print(self.model)

class Model_Step0(ModelManager):
    def Init_model(self):
        print("Model Init")
        self.create_model()

        if os.path.exists(self.model_path):
            print("\tLoad Pretrained Weights")
            self.load_model()

    def create_model(self, model_func=resnet18, default_weights=ResNet18_Weights.DEFAULT):
        print("\tCreate Model")
        model = model_func(weights=default_weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 5)  # 클래스 수에 맞게 마지막 레이어 변경
        self.model = model


    
