import torch
import matplotlib.pyplot as plt
import os

def train(model, epochs, LEARNING_RATE, loader, modelDir):

    loss_function = torch.nn.MSELoss()
    losses = []

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE
        )

    for epoch in range(epochs):
        # Armazena todos os valores de perda de uma unica época com
        # o objetivo de calcular a média.
        epoch_loss = []
        for (image, _) in loader:

            # Imagem original
            image = image.reshape(-1, 28 * 28)
            # Imagem reconstruída
            reconstructed = model(image)

            loss = loss_function(reconstructed, image)
            epoch_loss.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Calcula a média de todas as percas de uma época.
        avg_loss = sum(epoch_loss)/len(epoch_loss)
        losses.append(avg_loss)

        print(f'| Epoch: {epoch} Loss: {avg_loss}')

    # modelDir é uma array onde o primeiro item é o diretorio da pasta
    # e o segundo é o nome do arquivo.
    os.makedirs(modelDir[0], exist_ok = True) 
    pathFile = modelDir[0] + modelDir[1]

    torch.save(model.state_dict(), pathFile)

    plt.style.use('fivethirtyeight')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.show()