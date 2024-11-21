from collections import OrderedDict
import torch
from omegaconf import DictConfig
from model import Net, test


def get_on_fit_config(config: DictConfig):
    """Retorna uma função que prepara a configuração para enviar aos clientes."""

    def fit_config_fn(server_round: int):
        # Esta função será executada pela estratégia no método
        # `configure_fit()`.

        # Aqui, estamos retornando a mesma configuração em cada rodada,
        # mas você pode usar o argumento de entrada `server_round`
        # para adaptar essas configurações ao longo do tempo para os clientes.
        # Por exemplo, você pode desejar que os clientes utilizem uma
        # taxa de aprendizado diferente nas etapas posteriores do processo
        # de aprendizado federado (e.g., uma taxa de aprendizado menor após N rodadas).

        return {
            "lr": config.lr,  # Taxa de aprendizado
            "momentum": config.momentum,  # Momentum usado no otimizador
            "local_epochs": config.local_epochs,  # Número de épocas de treinamento local
        }

    return fit_config_fn


def get_evaluate_fn(num_classes: int, testloader):
    """Define a função para avaliação global no servidor."""

    def evaluate_fn(server_round: int, parameters, config):
        # Esta função é chamada pelo método `evaluate()` da estratégia
        # e recebe como argumentos de entrada o número atual da rodada e os
        # parâmetros do modelo global.
        # A função utiliza esses parâmetros e avalia o modelo global
        # em um conjunto de dados de teste/validação.

        model = Net(num_classes)  # Inicializa o modelo com o número de classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Carrega os parâmetros no modelo
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        # Avalia o modelo global no conjunto de teste.
        # Em configurações mais realistas, você realizaria esta avaliação
        # apenas ao final do experimento de aprendizado federado.
        # Você pode usar o argumento `server_round` para determinar se
        # esta é a última rodada. Caso contrário, utilize preferencialmente
        # um conjunto de validação global.
        loss, accuracy = test(model, testloader, device)

        # Relata a perda e qualquer outra métrica (dentro de um dicionário).
        # Neste caso, relatamos a acurácia global no teste.
        return loss, {"accuracy": accuracy}

    return evaluate_fn
