"""Testes dos algoritmos (rodar com: python -m pytest  ou  python tests/test_algoritmos.py)."""
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import dados, degradacao as dg, grafo, planejar_turno  # noqa: E402


def test_tres_versoes_iguais():
    random.seed(1)
    for n in range(0, 22):
        s = [random.random() * 3 for _ in range(n)]
        tab = dg.degradacao_tabulacao(s, .28, .17)
        for t in range(n):
            assert math.isclose(dg.degradacao_recursiva(s, t, .28, .17), tab[t], rel_tol=1e-12)
            assert math.isclose(dg.degradacao_memo(s, t, .28, .17, {}), tab[t], rel_tol=1e-12)


def test_chamadas_recursiva_fibonacci():
    # nº de chamadas da recursiva pura para D(n-1) = 2·F(n+2) − 1 (cresce como Fibonacci)
    fib = [0, 1]
    for _ in range(40):
        fib.append(fib[-1] + fib[-2])
    for n in range(1, 18):
        c = {}
        dg.degradacao_recursiva([1.0] * n, n - 1, .5, .3, c)
        assert c["chamadas"] == 2 * fib[n + 2] - 1


def test_bfs_dijkstra_astar():
    rede = grafo.carregar_rede(dados.PASTA_DADOS / "rede_estacoes.json")
    adj = grafo.montar_adjacencia(rede)
    coords = grafo.coordenadas(rede)
    vmax = rede["metodo"]["velocidade_maxima_kmh"]
    dist, pred, _ = grafo.dijkstra(adj, "BASE")
    saltos, _ = grafo.bfs(adj, "BASE")
    assert set(dist) == set(adj) == set(saltos)
    for destino in adj:
        custo, _, _ = grafo.a_estrela(adj, "BASE", destino, grafo.heuristica_tempo(coords, destino, vmax))
        assert math.isclose(custo, dist[destino], abs_tol=1e-9)
        rota = grafo.reconstruir_rota(pred, "BASE", destino)
        assert rota[0][0] == "BASE" and rota[-1][0] == destino


def test_heuristica_consistente():
    rede = grafo.carregar_rede(dados.PASTA_DADOS / "rede_estacoes.json")
    adj = grafo.montar_adjacencia(rede)
    coords = grafo.coordenadas(rede)
    vmax = rede["metodo"]["velocidade_maxima_kmh"]
    for destino in adj:
        h = grafo.heuristica_tempo(coords, destino, vmax)
        for u in adj:
            for v, w, _ in adj[u]:
                assert h(u) <= w + h(v) + 1e-9


def test_planejar_turno_respeita_horas():
    for horas in (2, 4, 6, 8, 12):
        plano = planejar_turno("BASE", horas)
        assert plano["resumo"]["tempo_usado_min"] <= horas * 60 + 1e-6
        posicoes = [v["posicao_ranking"] for v in plano["visitas"]]
        assert len(set(v["escada"] for v in plano["visitas"])) == len(posicoes)


if __name__ == "__main__":
    for nome, f in list(globals().items()):
        if nome.startswith("test_"):
            f()
            print("ok", nome)
