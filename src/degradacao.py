"""Degradação acumulada das escadas rolantes — três implementações da mesma recorrência.

Modelo
------
Cada escada tem uma sequência de leituras mensais. Cada leitura traz, para as cinco famílias de sinal do
desafio (vibração, temperatura, corrente/tensão, nível de óleo e velocidade), o número de ocorrências do mês
(derivado do histórico real de manutenção da ViaQuatro, ver notebook de EDA).

1. Severidade do mês t (combinação ponderada das cinco famílias):

       s(t) = Σ_f  w_f · x_f(t)

   w_f = duração média de indisponibilidade das falhas da família f ÷ média das cinco famílias
   (uma falha de vibração para a escada, em média, muito mais tempo que uma de óleo).

2. Degradação acumulada (recorrência de 2ª ordem, com memória de dois meses):

       D(t) = s(t) + α · D(t−1) + β · D(t−2),      D(−1) = D(−2) = 0

   α e β vêm do ajuste AR(2) da série mensal real (EDA, seção 5.3): o estado de um mês carrega parte do estado
   dos dois meses anteriores, e α + β < 1 garante que eventos antigos "esfriam" com o tempo.

Por ter DUAS chamadas recursivas, a versão recursiva pura repete subproblemas (como Fibonacci) e cresce
como O(φⁿ). A memoização e a tabulação resolvem cada D(t) uma única vez: O(n).
"""
from __future__ import annotations

import sys
import time
from typing import Callable, Dict, List, Optional, Sequence

FAMILIAS = ("vibracao", "temperatura", "corrente_tensao", "oleo", "velocidade")


def severidade(leitura: dict, pesos: Dict[str, float]) -> float:
    """Calcula a severidade de uma leitura mensal.

    Args:
        leitura: dicionário com a contagem de ocorrências de cada família no mês (chaves de ``FAMILIAS``).
        pesos: peso de cada família (mesmas chaves).

    Returns:
        Soma ponderada ``Σ w_f · x_f``. Famílias ausentes contam como zero.
    """
    return float(sum(pesos[f] * float(leitura.get(f, 0) or 0) for f in FAMILIAS))


def serie_severidade(leituras: Sequence[dict], pesos: Dict[str, float]) -> List[float]:
    """Converte a sequência de leituras (ordenada no tempo) na sequência de severidades ``s(0..n-1)``."""
    return [severidade(l, pesos) for l in leituras]


# --------------------------------------------------------------------------------------------
# 1) Recursiva pura
# --------------------------------------------------------------------------------------------
def degradacao_recursiva(s: Sequence[float], t: int, alfa: float, beta: float,
                         contador: Optional[Dict[str, int]] = None) -> float:
    """Degradação acumulada ``D(t)`` — versão recursiva pura (sem reaproveitar resultados).

    Args:
        s: severidades mensais ``s(0..n-1)``.
        t: índice do mês (``-1`` ou menor = antes da primeira leitura).
        alfa: peso da memória de 1 mês.
        beta: peso da memória de 2 meses.
        contador: dicionário opcional; ``contador["chamadas"]`` é incrementado a cada chamada (para análise).

    Returns:
        ``D(t)``.

    Complexidade: tempo O(φⁿ) (φ ≈ 1,618), memória O(n) de pilha. Inviável para séries longas.
    """
    if contador is not None:
        contador["chamadas"] = contador.get("chamadas", 0) + 1
    if t < 0:
        return 0.0
    return (s[t]
            + alfa * degradacao_recursiva(s, t - 1, alfa, beta, contador)
            + beta * degradacao_recursiva(s, t - 2, alfa, beta, contador))


# --------------------------------------------------------------------------------------------
# 2) Recursiva com memoização (dicionário)
# --------------------------------------------------------------------------------------------
def degradacao_memo(s: Sequence[float], t: int, alfa: float, beta: float,
                    memo: Optional[Dict[int, float]] = None,
                    contador: Optional[Dict[str, int]] = None) -> float:
    """Degradação acumulada ``D(t)`` — recursiva com memoização em dicionário (top-down).

    Args:
        s, t, alfa, beta: como em :func:`degradacao_recursiva`.
        memo: dicionário ``{t: D(t)}`` reaproveitado entre chamadas. Se ``None``, um novo é criado.
        contador: contagem opcional de chamadas.

    Returns:
        ``D(t)``.

    Complexidade: tempo O(n), memória O(n) (dicionário + pilha de recursão).

    Raises:
        RecursionError: se ``t`` passar do limite de recursão do Python (~1000 por padrão).
    """
    if memo is None:
        memo = {}
    if contador is not None:
        contador["chamadas"] = contador.get("chamadas", 0) + 1
    if t < 0:
        return 0.0
    if t in memo:
        return memo[t]
    memo[t] = (s[t]
               + alfa * degradacao_memo(s, t - 1, alfa, beta, memo, contador)
               + beta * degradacao_memo(s, t - 2, alfa, beta, memo, contador))
    return memo[t]


# --------------------------------------------------------------------------------------------
# 3) Iterativa com tabulação (bottom-up)
# --------------------------------------------------------------------------------------------
def degradacao_tabulacao(s: Sequence[float], alfa: float, beta: float) -> List[float]:
    """Degradação acumulada de TODOS os meses — iterativa com tabela (bottom-up).

    Args:
        s: severidades mensais ``s(0..n-1)``.
        alfa, beta: coeficientes da recorrência.

    Returns:
        Lista ``[D(0), D(1), ..., D(n-1)]``.

    Complexidade: tempo O(n), memória O(n) (a tabela). Sem recursão, portanto sem limite de pilha.
    """
    tabela = [0.0] * len(s)
    for t in range(len(s)):
        d1 = tabela[t - 1] if t >= 1 else 0.0
        d2 = tabela[t - 2] if t >= 2 else 0.0
        tabela[t] = s[t] + alfa * d1 + beta * d2
    return tabela


# --------------------------------------------------------------------------------------------
# Medição e ranking
# --------------------------------------------------------------------------------------------
def medir(funcao: Callable[[], object], repeticoes: int = 1) -> float:
    """Executa ``funcao`` ``repeticoes`` vezes e devolve o tempo médio por execução, em segundos."""
    inicio = time.perf_counter()
    for _ in range(repeticoes):
        funcao()
    return (time.perf_counter() - inicio) / repeticoes


def degradacao_final(s: Sequence[float], alfa: float, beta: float, versao: str = "tabulacao") -> float:
    """Degradação na última leitura usando a versão escolhida (``recursiva``, ``memo`` ou ``tabulacao``)."""
    if not s:
        return 0.0
    if versao == "recursiva":
        return degradacao_recursiva(s, len(s) - 1, alfa, beta)
    if versao == "memo":
        limite = sys.getrecursionlimit()
        if len(s) * 2 + 50 > limite:
            sys.setrecursionlimit(len(s) * 2 + 50)
        return degradacao_memo(s, len(s) - 1, alfa, beta, {})
    return degradacao_tabulacao(s, alfa, beta)[-1]


def ranking_criticidade(leituras_por_escada: Dict[str, Sequence[dict]], pesos: Dict[str, float],
                        alfa: float, beta: float, versao: str = "tabulacao") -> List[dict]:
    """Calcula a degradação acumulada final de cada escada e devolve o ranking (maior primeiro).

    Args:
        leituras_por_escada: ``{escada: [leitura_mes0, leitura_mes1, ...]}`` em ordem cronológica.
        pesos: peso de cada família.
        alfa, beta: coeficientes da recorrência.
        versao: implementação usada no cálculo.

    Returns:
        Lista de dicionários ``{"posicao", "escada", "degradacao", "familia_dominante", "contribuicao"}``,
        onde ``contribuicao`` mostra quanto cada família pesou na degradação final (explicabilidade).
    """
    linhas = []
    for escada, leituras in leituras_por_escada.items():
        s = serie_severidade(leituras, pesos)
        d = degradacao_final(s, alfa, beta, versao)
        # contribuição de cada família: a recorrência é linear, então D = Σ_f D_f
        contrib = {}
        for f in FAMILIAS:
            s_f = [pesos[f] * float(l.get(f, 0) or 0) for l in leituras]
            contrib[f] = degradacao_tabulacao(s_f, alfa, beta)[-1] if s_f else 0.0
        dominante = max(contrib, key=contrib.get) if d > 0 else "—"
        linhas.append({"escada": escada, "degradacao": d, "familia_dominante": dominante, "contribuicao": contrib})
    linhas.sort(key=lambda r: (-r["degradacao"], r["escada"]))
    for i, r in enumerate(linhas, 1):
        r["posicao"] = i
    return linhas
