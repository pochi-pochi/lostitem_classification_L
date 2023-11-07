import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import torch
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# 前処理
image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

# Datasetの作製
Data_folder = './dataset/train'
dataset = datasets.ImageFolder(root=Data_folder,
                               transform=image_transform)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
dataloaders_dict = {'train': train_loader, 'test': test_loader}


# モデルの定義
use_pretrained = True
net = models.vgg16(pretrained=use_pretrained)

net.classifier[6] = nn.Linear(in_features=4096, out_features=26)
net = net.to(device)

net.train()

print('ネットワークの訓練準備完了')


# 損失関数、最適化手法を定義
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.001)


# モデル学習用関数
def train_model(net, data_loaders_dist, criterion, optimizer, num_epochs):
    net.to(device)
    for epoch in range(num_epochs):
        print(f'{epoch+1}/{num_epochs}')
        print('----------------------------')

        for phase in ['train', 'test']:
            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels in tqdm(data_loaders_dist[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(data_loaders_dist[phase].dataset)
            epoch_acc = epoch_corrects.double(
                ) / len(data_loaders_dist[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))


# 学習の実行
num_epochs = 1
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)


# 画像の前処理関数
def preprocess_image(image_path):
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = image_transform(image).unsqueeze(0)  # バッチ次元の追加
    return image.to(device)


# 推論関数
def predict_image(model, image_path):
    model.eval()  # モデルを評価モードに設定
    image = preprocess_image(image_path)
    with torch.no_grad():  # 勾配計算を無効化
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
    return preds.item()


# クラス名の取得
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

# 推論の実行
image_path = './test_img/shiba.jpg'
pred_class_idx = predict_image(net, image_path)
pred_class_name = idx_to_class[pred_class_idx]
print(f'Predicted class index: {pred_class_idx}, name: {pred_class_name}')
