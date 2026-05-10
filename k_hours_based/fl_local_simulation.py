import flwr as fl
from flwr.common import Code, DisconnectRes, EvaluateRes, FitRes, GetParametersRes, GetPropertiesRes, Status
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import Server
from threading import Lock


class LocalNumPyClientProxy(ClientProxy):
    def __init__(self, cid: str, client_factory):
        super().__init__(cid)
        self._client_factory = client_factory
        self._client: fl.client.NumPyClient | None = None
        self._client_lock = Lock()

    def _get_client(self) -> fl.client.NumPyClient:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = self._client_factory(self.cid)
        return self._client

    def get_properties(self, ins, timeout, group_id):
        return GetPropertiesRes(status=Status(code=Code.OK, message=""), properties={})

    def get_parameters(self, ins, timeout, group_id):
        parameters = self._get_client().get_parameters(ins.config)
        return GetParametersRes(
            status=Status(code=Code.OK, message=""),
            parameters=fl.common.ndarrays_to_parameters(parameters),
        )

    def fit(self, ins, timeout, group_id):
        parameters = fl.common.parameters_to_ndarrays(ins.parameters)
        updated_parameters, num_examples, metrics = self._get_client().fit(parameters, ins.config)
        return FitRes(
            status=Status(code=Code.OK, message=""),
            parameters=fl.common.ndarrays_to_parameters(updated_parameters),
            num_examples=num_examples,
            metrics=metrics,
        )

    def evaluate(self, ins, timeout, group_id):
        parameters = fl.common.parameters_to_ndarrays(ins.parameters)
        loss, num_examples, metrics = self._get_client().evaluate(parameters, ins.config)
        return EvaluateRes(
            status=Status(code=Code.OK, message=""),
            loss=loss,
            num_examples=num_examples,
            metrics=metrics,
        )

    def reconnect(self, ins, timeout, group_id):
        return DisconnectRes(reason="ack")


def start_local_simulation(
    *,
    client_factory,
    num_clients: int,
    num_rounds: int,
    strategy: fl.server.strategy.Strategy,
    max_workers: int | None = None,
):
    client_manager = SimpleClientManager()
    for idx in range(num_clients):
        client_manager.register(LocalNumPyClientProxy(str(idx), client_factory))

    server = Server(client_manager=client_manager, strategy=strategy)
    server.max_workers = max_workers
    history, _ = server.fit(num_rounds=num_rounds, timeout=None)
    return history
