import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from modelClass import Autoencoder
from train import train
import os

model_filename = 'AutoencoderModel.pt'
model_Dir = ['./models/', model_filename]
BATCH_SIZE = 64
SHUFFLE = True
Learning_rate = 0.001
Epochs = 10

rootDir = './data'
Train = True
downloadDataset = True
tensor_transform = transforms.ToTensor()
 
dataset = datasets.MNIST(
    root = rootDir,
    train = Train,
    download = downloadDataset,
    transform = tensor_transform
    )

loader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size = BATCH_SIZE,
    shuffle = SHUFFLE
    )

model = Autoencoder()

# model_Dir é uma array onde o primeiro item é o diretório da pasta
# e o segundo é o nome do arquivo.
pathFile = model_Dir[0] + model_Dir[1]
isExist = os.path.exists(pathFile)
# Verifica se existe um modelo já treinado, caso contrário, inicializa o treino.
if isExist:
   model.load_state_dict(torch.load(pathFile))
else:
   train(model, Epochs, Learning_rate, loader, model_Dir)

# Armazena todas as imagens originais
imageList = []
# Armazena todas as imagens reconstruidas
reconstructedList = []

# Usa o modelo já treinado para reconstruir as imagens e salvar nas listas para exibir depois
def reconstructImages():
   for (image, _) in loader:
         image = image.reshape(-1, 28 * 28)
         reconstructed = model(image)
         imageList.append(image)
         reconstructedList.append(reconstructed.detach())

reconstructImages()

f, axarr = plt.subplots(1,2) 
plt.style.use('grayscale')

item = imageList[0].reshape(-1, 28, 28)
axarr[0].imshow(item[0])

item = reconstructedList[0].reshape(-1, 28, 28)
axarr[1].imshow(item[0])

plt.show()