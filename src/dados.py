"""Leitura dos arquivos de dados do projeto (pasta ``data/``), só com a biblioteca padrão."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

PASTA_DADOS = Path(__file__).resolve().parents[1] / "data"


def carregar_leituras(caminho: str | Path = PASTA_DADOS / "leituras_mensais_sensores.csv") -> Dict[str, List[dict]]:
    """Lê as leituras mensais e agrupa por escada, em ordem cronológica.

    Returns:
        ``{escada: [{"mes": "2023-01", "vibracao": 1, ..., "horas_paradas": 2.5}, ...]}``
    """
    por_escada: Dict[str, List[dict]] = defaultdict(list)
    with open(caminho, encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            reg = {k: (v if k in ("escada", "estacao", "mes") else float(v)) for k, v in linha.items()}
            por_escada[reg["escada"]].append(reg)
    for regs in por_escada.values():
        regs.sort(key=lambda r: r["mes"])
    return dict(por_escada)


def carregar_cadastro(caminho: str | Path = PASTA_DADOS / "cadastro_escadas.csv") -> Dict[str, dict]:
    """Lê o cadastro de escadas: ``{escada: {"estacao", "nome_estacao", "tempo_atendimento_h", ...}}``."""
    num = lambda v: float(v) if v not in ("", None) else 0.0
    with open(caminho, encoding="utf-8") as f:
        return {l["escada"]: {**l, "tempo_atendimento_h": num(l["tempo_atendimento_h"]),
                              "falhas_ano": num(l["falhas_ano"]), "horas_paradas_ano": num(l["horas_paradas_ano"])}
                for l in csv.DictReader(f)}


def carregar_parametros(caminho: str | Path = PASTA_DADOS / "parametros_degradacao.json") -> dict:
    """Lê ``alfa``, ``beta`` e ``pesos_familias`` estimados na EDA."""
    return json.loads(Path(caminho).read_text(encoding="utf-8"))
