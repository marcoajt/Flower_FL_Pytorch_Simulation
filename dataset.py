import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor

def get_mnist(data_path: str = "./data"):
    """Faz o download do conjunto de dados MNIST e aplica transformações mínimas."""
    
    # Definição das transformações: converte para tensor e normaliza os dados
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    
    # Faz o download dos dados de treino e teste do MNIST
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    
    return trainset, testset

def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """Faz o download do MNIST e gera partições IID (independente e identicamente distribuídas)."""
    
    # Faz o download do MNIST se ainda não estiver no sistema
    trainset, testset = get_mnist()

    # Divide o conjunto de treino em `num_partitions` partições (uma por cliente)
    # Calcula o número de exemplos de treino por partição
    num_images = len(trainset) // num_partitions

    # Cria uma lista com o tamanho de cada partição (todas do mesmo tamanho)
    partition_len = [num_images] * num_partitions

    # Divide os dados aleatoriamente. Cada partição terá `num_images` exemplos de treino
    # Observação: Esta é a forma mais simples de dividir o conjunto de dados.
    # Partições mais realistas podem introduzir heterogeneidade, como diferentes distribuições de classes
    # para cada cliente. Pesquise sobre Dirichlet (LDA) ou particionamento patológico em FL para explorar.
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # Cria DataLoaders para suporte ao treino e validação
    trainloaders = []
    valloaders = []

    # Para cada conjunto de treino, separa uma parte para validação
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        # Divide os dados em treino e validação
        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # Constrói os DataLoaders e adiciona às respectivas listas.
        # Dessa forma, o cliente i receberá o i-ésimo elemento das listas trainloaders e valloaders
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # O conjunto de teste permanece intacto (não é particionado).
    # Este conjunto de teste será mantido no servidor e usado para avaliar o desempenho
    # do modelo global após cada rodada. 
    # Nota: Em configurações mais realistas, seria ideal usar um conjunto de validação no servidor.
    # Em alguns cenários, principalmente fora da simulação, pode não ser viável construir um
    # conjunto de validação no lado do servidor.
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader
