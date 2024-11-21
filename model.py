import torch
import torch.nn as nn
import torch.nn.functional as F

# Observação: o modelo e as funções aqui definidas não possuem componentes específicos de FL (Aprendizado Federado).

class Net(nn.Module):
    """Uma CNN simples adequada para tarefas visuais simples."""
    
    def __init__(self, num_classes: int) -> None:
        super(Net, self).__init__()
        # Primeira camada de convolução: 1 canal de entrada, 6 filtros, kernel 5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # Camada de pooling: reduz as dimensões pela metade (kernel 2x2)
        self.pool = nn.MaxPool2d(2, 2)
        # Segunda camada de convolução: 6 canais de entrada, 16 filtros, kernel 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # Camada totalmente conectada (fully connected)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # Dimensão de entrada baseada na saída da convolução
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)  # Saída corresponde ao número de classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define o fluxo de dados pela rede."""
        x = self.pool(F.relu(self.conv1(x)))  # Convolução -> ReLU -> Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolução -> ReLU -> Pooling
        x = x.view(-1, 16 * 4 * 4)  # Achata o tensor para entrada na camada totalmente conectada
        x = F.relu(self.fc1(x))  # Primeira camada totalmente conectada -> ReLU
        x = F.relu(self.fc2(x))  # Segunda camada totalmente conectada -> ReLU
        x = self.fc3(x)          # Camada de saída (logits)
        return x

def train(net, trainloader, optimizer, epochs, device: str):
    """
    Treina a rede no conjunto de treino.

    Este é um loop de treinamento simples usando PyTorch.
    """
    criterion = torch.nn.CrossEntropyLoss()  # Define a função de perda (entropia cruzada)
    net.train()  # Coloca o modelo em modo de treino
    net.to(device)  # Move o modelo para o dispositivo especificado (CPU ou GPU)

    for _ in range(epochs):  # Executa o treinamento por um número de épocas
        for images, labels in trainloader:  # Itera sobre os batches do conjunto de treino
            images, labels = images.to(device), labels.to(device)  # Move os dados para o dispositivo
            optimizer.zero_grad()  # Zera os gradientes acumulados
            loss = criterion(net(images), labels)  # Calcula a perda
            loss.backward()  # Calcula os gradientes
            optimizer.step()  # Atualiza os pesos do modelo com base nos gradientes

def test(net, testloader, device: str):
    """
    Valida a rede no conjunto de teste e retorna a perda e a acurácia.
    """
    criterion = torch.nn.CrossEntropyLoss()  # Função de perda
    correct, loss = 0, 0.0  # Inicializa os contadores de perda e acertos
    net.eval()  # Coloca o modelo em modo de avaliação
    net.to(device)  # Move o modelo para o dispositivo especificado

    with torch.no_grad():  # Desabilita o cálculo de gradientes (economiza memória e acelera)
        for data in testloader:  # Itera sobre os batches do conjunto de teste
            images, labels = data[0].to(device), data[1].to(device)  # Move os dados para o dispositivo
            outputs = net(images)  # Faz a previsão com o modelo
            loss += criterion(outputs, labels).item()  # Calcula a perda acumulada
            _, predicted = torch.max(outputs.data, 1)  # Obtém a classe prevista
            correct += (predicted == labels).sum().item()  # Conta os acertos

    accuracy = correct / len(testloader.dataset)  # Calcula a acurácia
    return loss, accuracy
