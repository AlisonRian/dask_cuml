import cudf
import dask
import dask_cudf
from cuml.dask.datasets import make_classification
from cuml.dask.ensemble import RandomForestClassifier
from cuml.metrics import accuracy_score
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_ml.model_selection import train_test_split

@dask.delayed
def _to_cudf_dataframe(chunk):
    return cudf.DataFrame(chunk)

@dask.delayed
def _to_cudf_series(chunk):
    return cudf.Series(chunk, dtype="int32")

def main():
    # 1. Inicializa o Cluster Dask otimizado para GPUs
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(f"Painel de monitoramento (Dashboard): {client.dashboard_link}")

    # 2. Gera dados de classificacao diretamente nas GPUs
    print("Gerando dados sint?ticos nas GPUs...")
    n_features = 20
    X_arr, y_arr = make_classification(
        n_samples=300_000,
        n_features=n_features,
        n_classes=2,
        n_parts=4,  # Divide os dados em particoes no cluster
        random_state=42,
    )

    # Converte os blocos gerados (dask.array) para colecoes distribuidas dask_cudf,
    X_parts = [_to_cudf_dataframe(p) for p in X_arr.to_delayed().ravel()]
    y_parts = [_to_cudf_series(p) for p in y_arr.to_delayed().ravel()]

    meta_X = cudf.DataFrame({i: cudf.Series(dtype="float32") for i in range(n_features)})
    meta_y = cudf.Series(dtype="int32")

    X = dask_cudf.from_delayed(X_parts, meta=meta_X)
    y = dask_cudf.from_delayed(y_parts, meta=meta_y)

    # 3. Divisao entre dados de treino e teste de forma distribu?da
    print("Dividindo dados entre treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # 4. Inicializa o Classificador Random Forest do dask/cuML
    print("Inicializando o modelo Random Forest distribu?do...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # 5. Treina (Fit) o modelo utilizando todas as GPUs do cluster
    print("Treinando o modelo...")
    model.fit(X_train, y_train)

    # 6. Realiza previsoes nos dados de teste
    print("Realizando predi??es...")
    predictions = model.predict(X_test)

    # 7. Avalia a acuracia do modelo
    print("Avaliando resultados...")
    y_test_eval = y_test.compute()
    predictions_eval = predictions.compute()

    accuracy = accuracy_score(y_test_eval, predictions_eval)
    print(f"\n?? Treinamento Conclu?do!")
    print(f"?? Acur?cia do Classificador: {accuracy * 100:.2f}%")

    # Fecha a conexao do cliente e o cluster de forma limpa
    client.close()
    cluster.close()

if __name__ == "__main__":
    main()
