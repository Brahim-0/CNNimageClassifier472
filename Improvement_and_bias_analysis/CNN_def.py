import torch
import torch.nn as nn


class EmotionCNN0(nn.Module):
    def __init__(self):
        super(EmotionCNN0, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 4)  # 4 emotions

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EmotionCNN1(nn.Module):
    def __init__(self):
        super(EmotionCNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 3 * 3, 2048)
        self.fc2 = nn.Linear(2048, 4)  # 4 emotions

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))

        x = x.view(-1, 512 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EmotionCNN2(nn.Module):
    def __init__(self):
        super(EmotionCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, padding=2)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 3 * 3, 2048)
        self.fc2 = nn.Linear(2048, 4)  # 4 emotions

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))

        x = x.view(-1, 512 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EmotionCNN3(nn.Module):
    def __init__(self):
        super(EmotionCNN3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 4)  # 4 emotions

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EmotionCNN4(nn.Module):
    def __init__(self):
        super(EmotionCNN4, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #48X48
        self.dropout1 = nn.Dropout2d(0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(0.1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout2d(0.1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout2d(0.1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 3 * 3, 128)  # Adjusted the input size for batch size 32
        self.dropout5 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 4)  # 4 emotions

    def forward(self, x):
        x = self.dropout1(torch.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout2(torch.relu(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout3(torch.relu(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout4(torch.relu(self.conv4(x)))
        x = self.pool(x)

        x = x.view(-1, 256 * 3 * 3)  # Adjusted the view shape for batch size 32
        x = torch.relu(self.fc1(x))
        x = self.dropout5(x)
        x = self.fc2(x)
        return x


class EmotionCNN5(nn.Module):
    def __init__(self):
        super(EmotionCNN5, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=5, padding=2)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=5, padding=2)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=5, padding=2)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=5, padding=2)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=5, padding=2)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(256 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 4)  # 4 emotions

    def forward(self, x):
        x = torch.relu(self.conv1_1(x))
        x = torch.relu(self.conv1_2(x))
        x = self.pool(x) #24
        x = torch.relu(self.conv2_1(x))
        x = torch.relu(self.conv2_2(x))
        x = self.pool(x) #12
        x = torch.relu(self.conv3_1(x))
        x = torch.relu(self.conv3_2(x))
        x = torch.relu(self.conv3_3(x))
        x = self.pool(x)  #6
        x = torch.relu(self.conv4_1(x))
        x = torch.relu(self.conv4_2(x))
        x = torch.relu(self.conv4_3(x))
        x = self.pool(x)  #3

        x = x.view(-1, 256 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionCNN6(nn.Module):
    def __init__(self, num_classes=4):
        super(EmotionCNN6, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.dropout(torch.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn3(self.conv3(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn4(self.conv4(x)))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionCNN7(nn.Module):
    def __init__(self):
        super(EmotionCNN7, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(self.dropout(torch.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn3(self.conv3(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn4(self.conv4(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn5(self.conv5(x)))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Variants of E7:
class EmotionCNN7K(nn.Module):
    def __init__(self):
        super(EmotionCNN7K, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(self.dropout(torch.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn3(self.conv3(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn4(self.conv4(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn5(self.conv5(x)))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class EmotionCNN7L(nn.Module):
    def __init__(self):
        super(EmotionCNN7L, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4608, 512) # adjust this line
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(self.dropout(nn.functional.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.dropout(nn.functional.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.dropout(nn.functional.relu(self.bn3(self.conv3(x)))))
        x = self.pool(self.dropout(nn.functional.relu(self.bn4(self.conv4(x)))))

        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EmotionCNN8(nn.Module):
    def __init__(self):
        super(EmotionCNN8, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=7, padding=3)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=7, padding=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=7, padding=3)
        self.bn5 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self, x):
        x = self.pool(self.dropout(torch.relu(self.bn1(self.conv1(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn2(self.conv2(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn3(self.conv3(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn4(self.conv4(x)))))
        x = self.pool(self.dropout(torch.relu(self.bn5(self.conv5(x)))))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

