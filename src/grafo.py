"""Rede de estações como grafo e algoritmos de caminho: BFS, Dijkstra e A*.

Representação: **lista de adjacência** ``{vértice: [(vizinho, minutos, modo), ...]}``, não direcionada e
com possibilidade de várias arestas entre o mesmo par (ex.: metrô e viatura entre duas estações vizinhas).
"""
from __future__ import annotations

import heapq
import json
import math
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Adjacencia = Dict[str, List[Tuple[str, float, str]]]
Predecessores = Dict[str, Optional[Tuple[str, str]]]   # vértice -> (anterior, modo da aresta usada)


def carregar_rede(caminho: str | Path) -> dict:
    """Lê o JSON da rede (``data/rede_estacoes.json``) e devolve o dicionário bruto."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def montar_adjacencia(rede: dict, modos: Optional[Iterable[str]] = None) -> Adjacencia:
    """Monta a lista de adjacência a partir do JSON da rede.

    Args:
        rede: dicionário com ``vertices`` e ``arestas`` (ver :func:`carregar_rede`).
        modos: se informado, usa apenas arestas desses modos (ex.: ``{"viatura", "a_pe"}`` no turno noturno).

    Returns:
        Lista de adjacência não direcionada ``{v: [(vizinho, minutos, modo)]}``, com vizinhos ordenados
        por id (para que o BFS seja determinístico).
    """
    modos = set(modos) if modos else None
    adj: Adjacencia = {v["id"]: [] for v in rede["vertices"]}
    for a in rede["arestas"]:
        if modos and a["modo"] not in modos:
            continue
        adj[a["origem"]].append((a["destino"], float(a["minutos"]), a["modo"]))
        adj[a["destino"]].append((a["origem"], float(a["minutos"]), a["modo"]))
    for v in adj:
        adj[v].sort(key=lambda x: (x[0], x[1]))
    return adj


def coordenadas(rede: dict) -> Dict[str, Tuple[float, float]]:
    """Devolve ``{vértice: (lat, lon)}``."""
    return {v["id"]: (v["lat"], v["lon"]) for v in rede["vertices"]}


# ------------------------------------------------------------------------------------------------
def bfs(adj: Adjacencia, origem: str) -> Tuple[Dict[str, int], Predecessores]:
    """Busca em largura: caminho com o MENOR NÚMERO DE TRECHOS (ignora os minutos).

    Args:
        adj: lista de adjacência.
        origem: vértice de partida.

    Returns:
        ``(saltos, pred)``: nº mínimo de arestas até cada vértice alcançável e o predecessor de cada um.

    Complexidade: O(V + E).
    """
    saltos = {origem: 0}
    pred: Predecessores = {origem: None}
    fila = deque([origem])
    while fila:
        u = fila.popleft()
        for v, _, modo in adj[u]:
            if v not in saltos:
                saltos[v] = saltos[u] + 1
                pred[v] = (u, modo)
                fila.append(v)
    return saltos, pred


def dijkstra(adj: Adjacencia, origem: str, destino: Optional[str] = None
             ) -> Tuple[Dict[str, float], Predecessores, int]:
    """Dijkstra com fila de prioridade: caminho de MENOR TEMPO (soma dos minutos).

    Args:
        adj: lista de adjacência (pesos não negativos).
        origem: vértice de partida.
        destino: se informado, para assim que o destino é fechado (útil para comparar com o A*).

    Returns:
        ``(dist, pred, expandidos)``: menor tempo até cada vértice, predecessores e nº de vértices expandidos.

    Complexidade: O((V + E) log V).
    """
    dist = {origem: 0.0}
    pred: Predecessores = {origem: None}
    fechados = set()
    heap = [(0.0, origem)]
    expandidos = 0
    while heap:
        d, u = heapq.heappop(heap)
        if u in fechados:
            continue
        fechados.add(u)
        expandidos += 1
        if u == destino:
            break
        for v, w, modo in adj[u]:
            nd = d + w
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                pred[v] = (u, modo)
                heapq.heappush(heap, (nd, v))
    return dist, pred, expandidos


def reconstruir_rota(pred: Predecessores, origem: str, destino: str) -> List[Tuple[str, Optional[str]]]:
    """Reconstrói o caminho ``origem → destino`` seguindo os predecessores de trás para frente.

    Returns:
        Lista ``[(vértice, modo usado para chegar nele), ...]``; o primeiro item tem modo ``None``.
        Lista vazia se o destino não for alcançável.
    """
    if destino not in pred:
        return []
    rota = []
    atual: Optional[str] = destino
    while atual is not None:
        anterior = pred[atual]
        rota.append((atual, anterior[1] if anterior else None))
        atual = anterior[0] if anterior else None
    rota.reverse()
    return rota if rota[0][0] == origem else []


def formatar_rota(rota: List[Tuple[str, Optional[str]]]) -> str:
    """Texto legível da rota, ex.: ``BASE →(viatura) VSO →(metro) MBI``."""
    if not rota:
        return "(sem caminho)"
    partes = [rota[0][0]]
    for v, modo in rota[1:]:
        partes.append(f"→({modo}) {v}")
    return " ".join(partes)


# ------------------------------------------------------------------------------------------------
def haversine_km(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Distância em linha reta (km) entre dois pontos ``(lat, lon)``."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def heuristica_tempo(coords: Dict[str, Tuple[float, float]], destino: str, v_max_kmh: float,
                     custo_fixo_min: float = 0.0, alcance_max_km: Optional[float] = None):
    """Cria a heurística do A* (estimativa, em minutos, do tempo que falta até ``destino``).

    ``h(n) = D(n)/v_max·60  +  ⌈D(n)/L_max⌉ · c_min``

    onde D(n) é a distância em linha reta até o destino, ``v_max`` a maior velocidade efetiva da rede,
    ``c_min`` o menor custo fixo por trecho (parada do metrô, estacionar a viatura) e ``L_max`` o maior
    comprimento em linha reta de um trecho. Sem ``alcance_max_km``, a segunda parcela é zero.

    Por que é **admissível**: todo trecho (u,v) custa pelo menos ``dist(u,v)/v_max + c_min``; para cobrir a
    distância D(n) são necessários pelo menos ⌈D(n)/L_max⌉ trechos (cada um avança no máximo L_max). Logo o
    custo real restante é ≥ h(n).
    Por que é **consistente**: pela desigualdade triangular, D(u) ≤ D(v) + dist(u,v) e
    ⌈D(u)/L_max⌉ ≤ ⌈D(v)/L_max⌉ + 1, portanto h(u) ≤ w(u,v) + h(v).
    """
    alvo = coords[destino]
    cache: Dict[str, float] = {}

    def h(n: str) -> float:
        if n not in cache:
            d = haversine_km(coords[n], alvo)
            extra = math.ceil(d / alcance_max_km - 1e-12) * custo_fixo_min if alcance_max_km else 0.0
            cache[n] = d / v_max_kmh * 60.0 + extra
        return cache[n]
    return h


def parametros_heuristica(rede: dict, modos: Optional[Iterable[str]] = None) -> Tuple[float, float, float]:
    """Calcula ``(v_max, c_min, L_max)`` para :func:`heuristica_tempo` a partir dos modos ativos da rede.

    v_max = maior velocidade efetiva entre os modos usados; c_min = menor custo fixo por trecho;
    L_max = maior distância em linha reta de um trecho ativo.
    """
    modos = set(modos) if modos else set(rede["modos"])
    ativos = [a for a in rede["arestas"] if a["modo"] in modos]
    usados = {a["modo"] for a in ativos}
    v_max = max(rede["modos"][m]["v_efetiva_kmh"] for m in usados)
    c_min = min(rede["modos"][m]["custo_fixo_min"] for m in usados)
    l_max = max(a["distancia_km"] for a in ativos)
    return v_max, c_min, l_max


def a_estrela(adj: Adjacencia, origem: str, destino: str, h) -> Tuple[float, Predecessores, int]:
    """A* — Dijkstra guiado por uma heurística ``h(n)`` que estima o tempo restante até o destino.

    Args:
        adj: lista de adjacência.
        origem, destino: vértices.
        h: função heurística (admissível e consistente para garantir o ótimo).

    Returns:
        ``(custo, pred, expandidos)``: menor tempo, predecessores e nº de vértices expandidos.

    Complexidade: no pior caso igual à do Dijkstra; na prática expande menos vértices.
    """
    g = {origem: 0.0}
    pred: Predecessores = {origem: None}
    fechados = set()
    heap = [(h(origem), 0.0, origem)]
    expandidos = 0
    while heap:
        _, gu, u = heapq.heappop(heap)
        if u in fechados:
            continue
        fechados.add(u)
        expandidos += 1
        if u == destino:
            return gu, pred, expandidos
        for v, w, modo in adj[u]:
            ng = gu + w
            if ng < g.get(v, math.inf):
                g[v] = ng
                pred[v] = (u, modo)
                heapq.heappush(heap, (ng + h(v), ng, v))
    return math.inf, pred, expandidos
