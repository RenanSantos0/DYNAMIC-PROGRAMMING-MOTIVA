"""Plano de manutenção do turno: quais escadas atender, em que ordem e por qual caminho."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from . import dados, degradacao, grafo


def _contexto_padrao(modos: Optional[Iterable[str]] = None) -> dict:
    """Carrega dados, calcula o ranking (tabulação) e monta o grafo — usado quando nada é passado."""
    p = dados.carregar_parametros()
    ranking = degradacao.ranking_criticidade(dados.carregar_leituras(), p["pesos_familias"], p["alfa"], p["beta"])
    rede = grafo.carregar_rede(dados.PASTA_DADOS / "rede_estacoes.json")
    return {"ranking": ranking, "adj": grafo.montar_adjacencia(rede, modos), "cadastro": dados.carregar_cadastro()}


def planejar_turno(base: str, horas: float, *, ranking: Optional[List[dict]] = None,
                   adj: Optional[grafo.Adjacencia] = None, cadastro: Optional[Dict[str, dict]] = None,
                   modos: Optional[Iterable[str]] = None, limite_ranking: int = 30,
                   degradacao_minima: float = 0.0, inicio: str = "08:00") -> dict:
    """Monta o plano do turno a partir do ranking de criticidade e das rotas de menor tempo (Dijkstra).

    Regra (gulosa por criticidade, com aproveitamento de viagem):

    1. candidatas = as ``limite_ranking`` escadas mais críticas com degradação > ``degradacao_minima``;
    2. a partir da posição atual, escolhe a escada **mais crítica ainda não atendida** cujo
       ``deslocamento + atendimento + volta à base`` ainda cabe no tempo restante do turno;
    3. ao chegar na estação, atende também as outras candidatas **da mesma estação** (sem novo deslocamento),
       em ordem de criticidade, enquanto couber;
    4. repete até nenhuma candidata caber; termina com a volta à base.

    Assim, a escada mais crítica é sempre atendida primeiro (segurança) e o tempo de deslocamento
    não é desperdiçado. O retorno à base é sempre garantido dentro das ``horas`` do turno.

    Args:
        base: vértice de partida e chegada (ex.: ``"BASE"``).
        horas: duração do turno em horas.
        ranking: saída de :func:`degradacao.ranking_criticidade`. Se ``None``, é calculado a partir de ``data/``.
        adj: lista de adjacência da rede. Se ``None``, é montada a partir de ``data/rede_estacoes.json``.
        cadastro: ``{escada: {"estacao", "tempo_atendimento_h", ...}}``. Se ``None``, lido de ``data/``.
        modos: restringe os modos de deslocamento (ex.: ``{"viatura", "a_pe"}`` à noite, com o metrô fechado).
        limite_ranking: quantas escadas do topo do ranking entram como candidatas.
        degradacao_minima: escadas com degradação igual ou abaixo deste valor não são atendidas.
        inicio: horário de início do turno (``"HH:MM"``), usado só para exibir os horários.

    Returns:
        Dicionário com ``visitas`` (ordem, escada, estação, criticidade, rota, horários), ``retorno`` e ``resumo``.

    Raises:
        ValueError: se a base não existir na rede ou ``horas`` não for positivo.
    """
    if ranking is None or adj is None or cadastro is None:
        ctx = _contexto_padrao(modos)
        ranking = ranking if ranking is not None else ctx["ranking"]
        adj = adj if adj is not None else ctx["adj"]
        cadastro = cadastro if cadastro is not None else ctx["cadastro"]
    if base not in adj:
        raise ValueError(f"base '{base}' não existe na rede")
    if horas <= 0:
        raise ValueError("horas deve ser positivo")

    limite = horas * 60.0
    h0, m0 = map(int, inicio.split(":"))
    relogio = lambda minutos: f"{int((h0 * 60 + m0 + minutos) // 60) % 24:02d}:{int((h0 * 60 + m0 + minutos) % 60):02d}"

    cache: Dict[str, tuple] = {}
    def caminhos(origem: str):
        if origem not in cache:
            dist, pred, _ = grafo.dijkstra(adj, origem)
            cache[origem] = (dist, pred)
        return cache[origem]

    dist_base, _ = caminhos(base)
    candidatas = [r for r in ranking if r["degradacao"] > degradacao_minima][:limite_ranking]
    candidatas = [r for r in candidatas if cadastro.get(r["escada"], {}).get("estacao") in dist_base]

    pos, t = base, 0.0
    visitas, pendentes = [], list(candidatas)
    deslocamento_total = atendimento_total = 0.0

    def cabe(estacao: str, servico: float, desloc: float) -> bool:
        return t + desloc + servico + dist_base[estacao] <= limite + 1e-9

    while True:
        dist_pos, pred_pos = caminhos(pos)
        escolhida = None
        for r in pendentes:  # já em ordem de criticidade
            est = cadastro[r["escada"]]["estacao"]
            serv = cadastro[r["escada"]]["tempo_atendimento_h"] * 60
            if cabe(est, serv, dist_pos[est]):
                escolhida = r
                break
        if escolhida is None:
            break
        est = cadastro[escolhida["escada"]]["estacao"]
        desloc = dist_pos[est]
        rota = grafo.reconstruir_rota(pred_pos, pos, est)
        t += desloc
        deslocamento_total += desloc
        # atende a escolhida e, em seguida, as demais candidatas da mesma estação que couberem
        mesma_estacao = [escolhida] + [r for r in pendentes if r is not escolhida and cadastro[r["escada"]]["estacao"] == est]
        primeira = True
        for r in mesma_estacao:
            serv = cadastro[r["escada"]]["tempo_atendimento_h"] * 60
            if not cabe(est, serv, 0.0):
                continue
            visitas.append({
                "ordem": len(visitas) + 1, "escada": r["escada"], "estacao": est,
                "posicao_ranking": r["posicao"], "degradacao": round(r["degradacao"], 2),
                "familia_dominante": r["familia_dominante"],
                "rota": grafo.formatar_rota(rota) if primeira else f"{est} (mesma estação)",
                "deslocamento_min": round(desloc, 1) if primeira else 0.0,
                "chegada": relogio(t), "atendimento_min": round(serv, 1), "fim": relogio(t + serv),
            })
            t += serv
            atendimento_total += serv
            pendentes.remove(r)
            primeira = False
        pos = est

    dist_pos, pred_pos = caminhos(pos)
    volta = dist_pos[base]
    retorno = {"rota": grafo.formatar_rota(grafo.reconstruir_rota(pred_pos, pos, base)) if pos != base else base,
               "deslocamento_min": round(volta, 1), "chegada": relogio(t + volta)}
    t += volta
    deslocamento_total += volta
    soma_top = sum(r["degradacao"] for r in candidatas) or 1.0
    resumo = {
        "base": base, "horas_turno": horas, "inicio": inicio, "termino": relogio(t),
        "escadas_atendidas": len(visitas), "candidatas": len(candidatas),
        "tempo_usado_min": round(t, 1), "deslocamento_min": round(deslocamento_total, 1),
        "atendimento_min": round(atendimento_total, 1), "ocioso_min": round(limite - t, 1),
        "pct_degradacao_coberta": round(sum(v["degradacao"] for v in visitas) / soma_top * 100, 1),
    }
    return {"visitas": visitas, "retorno": retorno, "resumo": resumo}
