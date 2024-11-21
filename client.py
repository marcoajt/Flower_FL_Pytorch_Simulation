from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar

from model import Net, test, train

class FlowerClient(fl.client.NumPyClient):
    """Define um cliente Flower."""

    def __init__(self, trainloader, valloader, num_classes) -> None:
        super().__init__()

        # DataLoaders que apontam para os dados associados a este cliente
        self.trainloader = trainloader
        self.valloader = valloader

        # Modelo inicialmente com pesos aleatórios
        self.model = Net(num_classes)

        # Verifica se este cliente tem acesso a suporte de GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        """Recebe os parâmetros e os aplica ao modelo local."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        """Extrai os parâmetros do modelo e os retorna como uma lista de arrays numpy."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Treina o modelo recebido pelo servidor (parâmetros) usando os dados
        que pertencem a este cliente. Em seguida, envia-o de volta ao servidor.
        """
        # Copia os parâmetros enviados pelo servidor para o modelo local do cliente
        self.set_parameters(parameters)

        # Obtém elementos da configuração enviada pelo servidor. Note que ter uma configuração
        # enviada pelo servidor cada vez que um cliente precisa participar é um mecanismo simples,
        # mas poderoso, para ajustar esses hiperparâmetros durante o processo de aprendizado federado.
        # Por exemplo, talvez você queira que os clientes reduzam sua taxa de aprendizado após um número
        # de rodadas ou que façam mais épocas locais em estágios posteriores na simulação.
        # Você pode controlar isso personalizando o que passa para `on_fit_config_fn` ao
        # definir sua estratégia.
        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # Um otimizador padrão
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # Realiza o treinamento local. Esta função é idêntica ao que você pode
        # ter usado antes em projetos não federados. Para implementações mais avançadas
        # de aprendizado federado, você pode querer ajustá-la, mas, em geral, do ponto de vista
        # do cliente, o "treinamento local" pode ser visto como uma forma de "treinamento centralizado"
        # dado um modelo pré-treinado (ou seja, o modelo recebido do servidor).
        train(self.model, self.trainloader, optim, epochs, self.device)

        # Clientes Flower precisam retornar três argumentos: o modelo atualizado, o número
        # de exemplos no cliente (embora isso dependa um pouco da sua escolha de estratégia de agregação),
        # e um dicionário de métricas (aqui você pode adicionar quaisquer dados adicionais, mas estes
        # idealmente são estruturas de dados pequenas).
        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": accuracy}

def generate_client_fn(trainloaders, valloaders, num_classes):
    """Retorna uma função que pode ser usada pelo VirtualClientEngine
    para criar um FlowerClient com id de cliente `cid`.
    """
    def client_fn(cid: str):
        # Esta função será chamada internamente pelo VirtualClientEngine
        # Cada vez que o cliente de id `cid` for solicitado a participar na
        # simulação de aprendizado federado (seja para executar fit() ou evaluate()).

        # Retorna um FlowerClient normal que usará os DataLoaders de treino/validação
        # correspondentes ao id `cid` como seus dados locais.
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            valloader=valloaders[int(cid)],
            num_classes=num_classes,
        ).to_client()

    # Retorna a função para criar o cliente
    return client_fn
