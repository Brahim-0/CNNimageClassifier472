import torch
import torch.nn as nn
import torch.nn.functional as F

# class ImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         print("step")
#         images, labels = batch
#         out = self(images)  # call forward
#         loss = F.cross_entropy(out, labels)  # Calculate loss
#         return loss
#
#     def validation_step(self, batch):
#         images, labels = batch
#         out = self(images)  # call forward
#         loss = F.cross_entropy(out, labels)  # Calculate loss
#         acc = accuracy(out, labels)  # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}
#
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
#
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch, result['train_loss'], result['val_loss'], result['val_acc']))
#
#
# def accuracy(outputs, labels):
#     _, preds = torch.max(outputs, dim=1)
#     return torch.tensor(torch.sum(preds == labels).item() / len(preds))
#
#
# class EMModel(ImageClassificationBase):
#     def __init__(self):
#         super(ImageClassificationBase, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(128 * 6 * 6, 512)
#         self.fc2 = nn.Linear(512, 4)  # 4 emotions
#
#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))
#         x = x.view(-1, 128 * 6 * 6)
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# Define CNN architecture aka 64
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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)  # Changed kernel_size to 5 and padding to 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)  # Changed kernel_size to 5 and padding to 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)  # Changed kernel_size to 5 and padding to 2
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, padding=2)  # Changed kernel_size to 5 and padding to 2
        self.conv5 = nn.Conv2d(256, 512, kernel_size=5, padding=2)  # Changed kernel_size to 5 and padding to 2

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
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
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
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(512 * 3 * 3, 128)
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

        x = x.view(-1, 512 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


