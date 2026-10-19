"""Gera data/rede_estacoes.json — rede de estações da Linha 4-Amarela (ViaQuatro/Motiva) + base de manutenção.

Vértices: as 11 estações da Linha 4 + o Terminal Vila Sônia (os 12 locais com escadas rolantes no
PMP-6ECR-0001, item 10.2) e 1 base de manutenção junto ao pátio de Vila Sônia.
Arestas: trechos de deslocamento, com peso em minutos, em três modos:

* metro   — entre estações vizinhas da linha: distância / 32 km/h (velocidade comercial) + 1 min de parada;
* a_pe    — integração Vila Sônia ↔ Terminal: distância / 4,5 km/h + 1 min;
* viatura — deslocamento por rua: distância em linha reta × 1,4 (fator de sinuosidade urbana) / 18 km/h + 5 min
            (estacionar e acessar a estação). Existe entre vizinhas (para turnos noturnos, com a linha fechada),
            da base às estações próximas e em alguns atalhos rodoviários.

As coordenadas são aproximadas (precisão de ~100 m) e servem para calcular as distâncias e a heurística do A*.
Os minutos são arredondados PARA CIMA em 0,1 min, garantindo que nenhuma aresta fique abaixo do tempo
"em linha reta na velocidade máxima" (condição usada para provar que a heurística do A* é admissível).

Uso: python scripts/gerar_rede.py
"""
import json
import math
from pathlib import Path

ESTACOES = [  # id, nome, lat, lon, nº de escadas no PMP
    ("LUZ", "Luz", -23.5363, -46.6352, 20),
    ("REP", "República", -23.5442, -46.6428, 5),
    ("HIG", "Higienópolis-Mackenzie", -23.5475, -46.6520, 26),
    ("PTA", "Paulista", -23.5553, -46.6620, 18),
    ("FRE", "Oscar Freire", -23.5617, -46.6722, 20),
    ("FRA", "Fradique Coutinho", -23.5660, -46.6845, 10),
    ("FAL", "Faria Lima", -23.5670, -46.6930, 10),
    ("PIN", "Pinheiros", -23.5672, -46.7019, 32),
    ("BUT", "Butantã", -23.5718, -46.7085, 14),
    ("MBI", "São Paulo-Morumbi", -23.5865, -46.7238, 13),
    ("VSO", "Vila Sônia", -23.5960, -46.7385, 20),
    ("TVS", "Terminal Vila Sônia", -23.5967, -46.7395, 9),
]
BASE = ("BASE", "Base de Manutenção (Pátio Vila Sônia)", -23.5985, -46.7430, 0)

V_METRO, V_VIATURA, V_PE = 32.0, 18.0, 4.5   # km/h
SINUOSIDADE, ACESSO_VIATURA = 1.4, 5.0


def haversine_km(a, b):
    """Distância em km entre dois pontos (lat, lon) sobre a esfera terrestre."""
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def teto(x):
    return math.ceil(x * 10) / 10


def minutos(modo, km):
    if modo == "metro":
        return teto(km / V_METRO * 60 + 1)
    if modo == "a_pe":
        return teto(km / V_PE * 60 + 1)
    return teto(km * SINUOSIDADE / V_VIATURA * 60 + ACESSO_VIATURA)


def main():
    nos = {e[0]: {"id": e[0], "nome": e[1], "tipo": "estacao", "linha": "4-Amarela", "lat": e[2], "lon": e[3], "escadas_pmp": e[4]}
           for e in ESTACOES}
    nos[BASE[0]] = {"id": BASE[0], "nome": BASE[1], "tipo": "base", "linha": "4-Amarela", "lat": BASE[2], "lon": BASE[3], "escadas_pmp": 0}
    pos = {k: (v["lat"], v["lon"]) for k, v in nos.items()}

    linha = [e[0] for e in ESTACOES if e[0] != "TVS"]
    pares = []
    for a, b in zip(linha, linha[1:]):
        pares += [(a, b, "metro"), (a, b, "viatura")]
    pares += [("VSO", "TVS", "a_pe"), ("VSO", "TVS", "viatura")]
    pares += [("BASE", "VSO", "viatura"), ("BASE", "TVS", "viatura"), ("BASE", "MBI", "viatura"),
              ("BASE", "BUT", "viatura"), ("BASE", "PIN", "viatura"), ("BASE", "PTA", "viatura")]
    pares += [("PIN", "FRA", "viatura"), ("FAL", "PTA", "viatura"), ("PTA", "LUZ", "viatura"),
              ("HIG", "LUZ", "viatura"), ("MBI", "PIN", "viatura"), ("BUT", "FRE", "viatura")]

    arestas = [{"origem": a, "destino": b, "modo": m, "distancia_km": round(haversine_km(pos[a], pos[b]), 3),
                "minutos": minutos(m, haversine_km(pos[a], pos[b]))} for a, b, m in pares]
    rede = {
        "descricao": "Rede de deslocamento da manutenção — Linha 4-Amarela (ViaQuatro/Motiva). Arestas não direcionadas, peso em minutos.",
        "metodo": {"metro": f"dist/{V_METRO} km/h*60 + 1 min", "a_pe": f"dist/{V_PE} km/h*60 + 1 min",
                   "viatura": f"dist*{SINUOSIDADE}/{V_VIATURA} km/h*60 + {ACESSO_VIATURA} min",
                   "velocidade_maxima_kmh": V_METRO, "arredondamento": "para cima, 0,1 min",
                   "coordenadas": "aproximadas (~100 m)"},
        "modos": {"metro": {"v_efetiva_kmh": V_METRO, "custo_fixo_min": 1.0},
                  "a_pe": {"v_efetiva_kmh": V_PE, "custo_fixo_min": 1.0},
                  "viatura": {"v_efetiva_kmh": round(V_VIATURA / SINUOSIDADE, 4), "custo_fixo_min": ACESSO_VIATURA}},
        "base": "BASE",
        "vertices": list(nos.values()),
        "arestas": arestas,
    }
    destino = Path(__file__).resolve().parents[1] / "data" / "rede_estacoes.json"
    destino.write_text(json.dumps(rede, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(nos)} vértices, {len(arestas)} arestas → {destino}")


if __name__ == "__main__":
    main()
