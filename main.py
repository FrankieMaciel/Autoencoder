import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from modelClass import Autoencoder
import os

model_filename = 'AutoencoderModel.pt'
model_Dir = './' + model_filename
BATCH_SIZE = 64
SHUFFLE = True
LEARNING_RATE = 0.001

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

isExist = os.path.exists(model_Dir)
if isExist:
   model.load_state_dict(torch.load(model_Dir))

loss_function = torch.nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = LEARNING_RATE
    )

epochs = 10
losses = []

def train():
   for epoch in range(epochs):
      for (image, _) in loader:

         # Imagem original
         image = image.reshape(-1, 28 * 28)
         # Imagem reconstruída
         reconstructed = model(image)

         loss = loss_function(reconstructed, image)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()
         losses.append(loss.detach())

      print('| Epoch: {}'.format(epoch))

   torch.save(model.state_dict(), model_filename)

   plt.style.use('fivethirtyeight')
   plt.xlabel('Iterations')
   plt.ylabel('Loss')
   plt.plot(losses[-100:])
   plt.show()

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

if isExist == False:
   train()

reconstructImages()

f, axarr = plt.subplots(1,2) 
plt.style.use('grayscale')

item = imageList[0].reshape(-1, 28, 28)
axarr[0].imshow(item[0])

item = reconstructedList[0].reshape(-1, 28, 28)
axarr[1].imshow(item[0])

plt.show()