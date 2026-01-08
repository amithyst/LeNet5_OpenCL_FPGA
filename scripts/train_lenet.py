import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time

# ==========================================
# 1. 定义 LeNet-5 模型 (ReLU 版)
# ==========================================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # C1: 1 -> 6, 5x5
        self.c1 = nn.Conv2d(1, 6, 5)
        # S2: MaxPool 2x2 (stride 2)
        self.s2 = nn.MaxPool2d(2, 2)
        # C3: 6 -> 16, 5x5
        self.c3 = nn.Conv2d(6, 16, 5)
        # S4: MaxPool 2x2
        self.s4 = nn.MaxPool2d(2, 2)
        # C5: 16 -> 120, 5x5
        self.c5 = nn.Conv2d(16, 120, 5)
        # F6: 120 -> 84
        self.f6 = nn.Linear(120, 84)
        # Output: 84 -> 10
        self.out = nn.Linear(84, 10)

    def forward(self, x):
        # 修改点: tanh -> relu
        x = torch.relu(self.c1(x))
        x = self.s2(x)
        x = torch.relu(self.c3(x))
        x = self.s4(x)
        x = torch.relu(self.c5(x))
        x = x.view(-1, 120) # Flatten
        x = torch.relu(self.f6(x))
        x = self.out(x) # 最后一层通常不需要激活，或者在 Host 做 Softmax
        return x

# ==========================================
# 2. 权重导出函数
# ==========================================
def save_bin(filename, tensor):
    output_dir = "../weights"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    path = os.path.join(output_dir, filename)
    data = tensor.detach().cpu().numpy().astype(np.float32)
    data.tofile(path)

def export_weights(model):
    print("\n>>> Exporting weights to '../weights/'...")
    save_bin("c1_weight.bin", model.c1.weight)
    save_bin("c1_bias.bin",   model.c1.bias)
    save_bin("c3_weight.bin", model.c3.weight)
    save_bin("c3_bias.bin",   model.c3.bias)
    save_bin("c5_weight.bin", model.c5.weight)
    save_bin("c5_bias.bin",   model.c5.bias)
    save_bin("f6_weight.bin", model.f6.weight)
    save_bin("f6_bias.bin",   model.f6.bias)
    save_bin("out_weight.bin", model.out.weight)
    save_bin("out_bias.bin",   model.out.bias)
    print(">>> Export Complete.")

# ==========================================
# 3. 训练与验证主逻辑
# ==========================================
def train_and_export():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理 (Resize 28 -> 32)
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    BATCH_SIZE = 64
    EPOCHS = 5 # ReLU 收敛很快，5轮通常足够

    print("Loading Datasets...")
    trainset = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    
    testset = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0

    print(f"Start Training (ReLU Version)...")
    
    for epoch in range(EPOCHS):
        model.train()
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test
        print(f"[Epoch {epoch+1}] Val Acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            export_weights(model)

    print(f"Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    train_and_export()