import pickle
from pathlib import Path
import flwr as fl
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import logging

from client import generate_client_fn
from dataset import prepare_dataset
from server import get_evaluate_fn, get_on_fit_config


# Um decorador para o Hydra. Indica que ele carregará o arquivo de configuração conf/base.yaml por padrão.
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Interpretar a configuração e obter o diretório de saída do experimento
    print(OmegaConf.to_yaml(cfg))
    # O Hydra cria automaticamente um diretório para os experimentos.
    # Por padrão, ele estará em <diretório atual>/outputs/<data>/<hora>.
    # Você pode recuperar o caminho para salvar os resultados da simulação.
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Preparar o conjunto de dados
    # Ao simular execuções de aprendizado federado (FL), temos muita liberdade sobre como os clientes do FL se comportam,
    # que dados possuem, quanto de dados possuem, etc. Isso não é possível em configurações reais de FL.
    #
    # Neste tutorial, vamos particionar o conjunto de dados MNIST em 100 clientes (padrão na configuração)
    # usando uma amostragem independente e identicamente distribuída (IID).
    # Este é o método mais simples de particionamento e adequado para este tutorial introdutório.
    trainloaders, validationloaders, testloader = prepare_dataset(
        cfg.num_clients, cfg.batch_size
    )

    ## 3. Definir os clientes
    # Na simulação, não queremos iniciar os clientes manualmente.
    # Delegamos isso para o VirtualClientEngine, que cria clientes quando necessário.
    client_fn = generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Definir a estratégia
    # Uma estratégia do Flower orquestra o pipeline de FL. Aqui usamos FedAvg,
    # que simplesmente calcula a média dos modelos recebidos dos clientes após o treinamento.
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.0,  # Em simulação, usamos min_fit_clients para controlar quantos clientes participam.
        min_fit_clients=cfg.num_clients_per_round_fit,  # Número de clientes para treinamento (fit).
        fraction_evaluate=0.0,  # Similar ao fraction_fit, mas para avaliação.
        min_evaluate_clients=cfg.num_clients_per_round_eval,  # Número de clientes para avaliação.
        min_available_clients=cfg.num_clients,  # Total de clientes disponíveis na simulação.
        on_fit_config_fn=get_on_fit_config(cfg.config_fit),  # Função para configurar o treinamento local dos clientes.
        evaluate_fn=get_evaluate_fn(cfg.num_classes, testloader),  # Função para avaliar o modelo global no servidor.
    )

    ## 5. Iniciar a simulação
    # Inicia a simulação com o conjunto de dados particionado, a função de cliente e a estratégia configurada.
    history = fl.simulation.start_simulation(
        client_fn=client_fn,  # Função para criar um cliente específico.
        num_clients=cfg.num_clients,  # Número total de clientes na simulação.
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),  # Configuração do servidor, incluindo o número de rodadas.
        strategy=strategy,  # Estratégia definida (FedAvg).
        client_resources={"num_cpus": 2, "num_gpus": 0.0},  # Controle de paralelismo da simulação.
    )

    ## 6. Salvar os resultados
    # Salva os resultados no diretório criado pelo Hydra no início do experimento.
    results_path = Path(save_path) / "results.pkl"
    results = {"history": history}
    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)

    # Configuração básica do logging
    logging.basicConfig(
        filename="app.log",  # Nome do arquivo de log
        filemode="w",        # 'w' sobrescreve, 'a' anexa
        level=logging.DEBUG, # Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Aplicação iniciada")  # Mensagem de exemplo registrada.


if __name__ == "__main__":
    main()
