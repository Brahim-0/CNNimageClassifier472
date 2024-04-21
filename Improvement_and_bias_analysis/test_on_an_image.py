import torch
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from CNN_def import EmotionCNN4

# Individual image prediction function
def predict_emotion(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = ToTensor()(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        emotion = dataset.classes[predicted.item()]
    return emotion


# define the dir for the dataset
data_dir = './dataset'
dataset = ImageFolder(data_dir, transform=ToTensor())

# Test the best model
model = EmotionCNN4()
model.load_state_dict(torch.load('best_model_E4_63.pt'))
model.eval()
print(predict_emotion("brightened_1058.jpg"))
