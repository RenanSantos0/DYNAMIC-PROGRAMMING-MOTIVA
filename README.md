# SensEscada · Dynamic Programming

**Challenge Motiva + FIAP 2026/2027 — Sprint 1 e 2**
Priorização da manutenção das 197 escadas rolantes da **ViaQuatro (Linha 4-Amarela)** com programação dinâmica e algoritmos em grafos.

| Integrante | RM |
|---|---|
| Felipe Estevo Santos | RM567780 |
| Igor Grave Teixeira | RM567663 |
| Renan dos Reis Santos | RM568540 |

---

## O que este repositório entrega

| Requisito | Onde está |
|---|---|
| Degradação acumulada por escada — **recursiva pura, memoizada (dict) e tabulação**, medição das três e **ranking por criticidade** | `src/degradacao.py` · notebook, seções 1 e 2 |
| Rede de estações como grafo (**lista de adjacência**), ≥ 8 estações da Motiva + 1 base, escadas ligadas às estações | `data/rede_estacoes.json`, `src/grafo.py` · seção 3 |
| **BFS** (menos trechos), **Dijkstra** (menor tempo), **reconstrução e exibição das rotas** | `src/grafo.py` · seção 3.2 |
| **A\*** com heurística justificada, comparado ao Dijkstra | `src/grafo.py` · seção 4 |
| **`planejar_turno(base, horas)`**: quais escadas, em que ordem, por qual caminho | `src/planejamento.py` · seção 5 |
| Análise de desempenho e interpretação | notebook, seções 1.3, 4.3, 6 e este README |
| Notebook executado do início ao fim | `notebook.ipynb` |

## Como executar

```bash
pip install -r requirements.txt     # pandas, numpy, matplotlib
jupyter notebook notebook.ipynb     # ou abrir no Google Colab / VS Code
python tests/test_algoritmos.py     # testes (também roda com pytest)
```

Uso direto da função de planejamento:

```python
from src import planejar_turno
plano = planejar_turno("BASE", 6)            # turno de 6 h saindo da base de manutenção
for v in plano["visitas"]:
    print(v["ordem"], v["escada"], v["chegada"], v["rota"])
print(plano["resumo"])
```

## Estrutura

```
├── notebook.ipynb                    # notebook executado (todas as saídas)
├── notebook.py                       # mesmo conteúdo em formato texto (diff legível no Git)
├── src/
│   ├── degradacao.py                 # severidade, 3 versões da degradação, medição, ranking
│   ├── grafo.py                      # lista de adjacência, BFS, Dijkstra, A*, reconstrução de rotas
│   ├── planejamento.py               # planejar_turno(base, horas)
│   └── dados.py                      # leitura dos CSV/JSON
├── data/
│   ├── leituras_mensais_sensores.csv # 201 escadas × 45 meses × 5 famílias de sinal (dados reais)
│   ├── cadastro_escadas.csv          # escada → estação, tempo típico de atendimento
│   ├── parametros_degradacao.json    # α, β e pesos (estimados na EDA)
│   └── rede_estacoes.json            # 13 vértices, 34 arestas (minutos)
├── scripts/gerar_rede.py             # gera a rede (método de cálculo dos minutos)
└── tests/test_algoritmos.py
```

## Dados

O dataset oficial da Motiva é o **histórico de manutenção do SAP** da ViaQuatro (2009–2026: 104 mil notas, 269 mil apontamentos de mão de obra, 27 mil movimentos de material). Ele **não traz leituras contínuas de sensor**.
No notebook de EDA (entrega de Data Science), cada falha real foi classificada pela família de sinal que poderia detectá-la (**vibração, temperatura, corrente/tensão, nível de óleo, velocidade**).
A "leitura" mensal de cada escada é o número de ocorrências de cada família no mês, na janela homogênea **jan/2023–set/2026** (a forma de registro das falhas mudou em 2023).

| Arquivo | Colunas principais |
|---|---|
| `leituras_mensais_sensores.csv` | `escada, estacao, mes, vibracao, temperatura, corrente_tensao, oleo, velocidade, horas_paradas` |
| `cadastro_escadas.csv` | `escada, estacao, nome_estacao, tempo_atendimento_h, falhas_ano, horas_paradas_ano` |
| `parametros_degradacao.json` | `alfa = 0.281`, `beta = 0.17`, `pesos_familias` |
| `rede_estacoes.json` | `vertices[id, nome, tipo, lat, lon, escadas_pmp]`, `arestas[origem, destino, modo, distancia_km, minutos]`, `modos` |

Quando houver leituras reais de sensores, basta gerar um CSV no mesmo formato: nenhuma função precisa mudar.

## 1. Degradação acumulada (programação dinâmica)

**Severidade do mês:** `s(t) = Σ w_f · x_f(t)`. O peso `w_f` é a duração média de indisponibilidade da família ÷ a média das cinco. Fica vibração 2,36 · temperatura 1,33 · corrente/tensão 0,72 · velocidade 0,34 · óleo 0,24.

**Recorrência:** `D(t) = s(t) + α·D(t−1) + β·D(t−2)`, com `D(−1) = D(−2) = 0`.
α = 0,281 e β = 0,170 vêm do ajuste AR(2) da série mensal real de falhas (autocorrelação de 0,33 no lag 1 e 0,26 no lag 2). Como α + β < 1, a degradação reflete a condição **recente**.
As duas chamadas recursivas criam **subproblemas sobrepostos**, como em Fibonacci.

| Versão | Função | Tempo | Memória | Resultado medido |
|---|---|---|---|---|
| Recursiva pura | `degradacao_recursiva` | O(φⁿ) | O(n) pilha | chamadas = 2·F(n+2) − 1; n = 28 → 1,7 milhão de chamadas (~0,1 s); 45 meses ≈ 5,9 bilhões de chamadas (≈ 5 min por escada, **≈ 17 h para as 201**) |
| Memoizada (dict) | `degradacao_memo` | O(n) | O(n) + pilha | microssegundos, mas `RecursionError` a partir de ~1.000 leituras |
| Tabulação | `degradacao_tabulacao` | O(n) | O(n) | 201 escadas × 45 meses em ~8 ms; 100 mil leituras em ~11 ms |

As três dão o mesmo resultado (diferença 0,0 em todas as escadas). **A tabulação é a versão usada em produção.**

**Ranking (set/2026), top 5:** REP-ER59 · LUZ-ER11 · PIN-ER24 · PTA-ER09 · PIN-ER21.
O ranking é explicável: como a recorrência é linear, D separa-se na contribuição de cada família. Vibração domina o topo; Butantã aparece por corrente/tensão e PIN-ER21 por temperatura.
Por estação, Butantã, Luz, República e Pinheiros têm a maior degradação média, o mesmo padrão encontrado na EDA.

## 2. Rede de estações, BFS e Dijkstra

- **Vértices (13):** Luz, República, Higienópolis-Mackenzie, Paulista, Oscar Freire, Fradique Coutinho, Faria Lima, Pinheiros, Butantã, São Paulo-Morumbi, Vila Sônia, Terminal Vila Sônia (os 12 locais com escadas no plano PMP-6ECR-0001) e a **base de manutenção** junto ao pátio de Vila Sônia.
- **Arestas (34)** com peso em minutos: `metro` (vizinhas, distância/32 km/h + 1 min), `a_pe` (VSO↔TVS) e `viatura` (distância × 1,4 / 18 km/h + 5 min; entre vizinhas, da base e atalhos). As coordenadas são aproximadas.
- **Cada escada** está ligada à estação do seu código (ex.: `PIN-ER24` → Pinheiros); as 201 escadas do dataset estão no grafo.

| Destino a partir da base | BFS (trechos / min) | Dijkstra (trechos / min) | Economia |
|---|---|---|---|
| Paulista | 1 / 49,6 | 8 / 32,5 | 17,1 min |
| Luz | 2 / 70,8 | 11 / 42,2 | 28,6 min (−40%) |
| Pinheiros | 1 / 30,5 | 4 / 20,0 | 10,5 min |

O BFS escolhe atalhos de viatura (menos trechos, mais lentos); o Dijkstra combina viatura até Vila Sônia com metrô. Para o técnico importa o tempo, por isso o **Dijkstra alimenta o planejamento**.

## 3. A* e a heurística

`h(n) = D(n)/v_máx · 60 + ⌈D(n)/L_máx⌉ · c_mín`

D é a distância em linha reta até o destino, v_máx a maior velocidade efetiva dos modos ativos (metrô 32 km/h de dia; à noite, viatura ≈ 12,9 km/h), c_mín o menor custo fixo por trecho e L_máx o maior trecho em linha reta.

- **Admissível:** todo trecho custa ≥ dist/v_máx + c_mín, e cobrir D exige ≥ ⌈D/L_máx⌉ trechos. A heurística nunca superestima.
- **Consistente:** pela desigualdade triangular, h(u) ≤ w(u,v) + h(v). Verificado numericamente: 0 violações em todas as arestas e destinos, de dia e de noite.

| Cenário | Expansões A\* vs Dijkstra | Tempo |
|---|---|---|
| Linha 4, diurna | praticamente iguais (rede em linha, heurística frouxa) | A\* ~2× mais lento (µs) |
| Linha 4, noturna | **19–32% menos** | ainda µs |
| Malha 3.600 vértices, noturna | **58% menos** | **~10% mais rápido** |
| Malha 3.600 vértices, diurna | 20% menos | mais lento (custo da heurística) |

O A\* sempre devolveu o **mesmo custo e a mesma rota** do Dijkstra. Ele compensa quando a heurística é justa e a rede é grande: é o caso de expandir para todas as linhas da Motiva e do planejamento noturno por viatura.

## 4. `planejar_turno(base, horas)`

1. Candidatas: as 30 escadas mais críticas do ranking.
2. Vai para a **mais crítica ainda não atendida** cujo deslocamento (Dijkstra) + atendimento + volta à base cabe no tempo restante.
3. Na estação, atende também as outras candidatas da **mesma estação**, sem novo deslocamento.
4. Repete e volta à base. O retorno é sempre garantido dentro das horas.

O tempo de atendimento de cada escada é a mediana real (horas de mão de obra ÷ técnicos) dos reparos daquela escada.

**Resultado — turno diurno de 6 h (08:00):** BASE → REP-ER59 (nº 1) → LUZ-ER11 (nº 2) → LUZ-ER13, LUZ-ER12, LUZ-ER09 (mesma estação) → BASE às 13:44.
São 5 escadas, 84 min de deslocamento (24%) e 26% da degradação do top 30 atendida.
**Turno noturno (só viatura):** 4 escadas, 152 min de deslocamento (43% do turno).
Checagens automáticas: cabe no turno, ordem por criticidade, mais crítica primeiro, sem repetição.
**Hipótese para validação:** duas equipes (trecho norte e trecho sul) cobririam mais risco com menos deslocamento.

## Limitações

- As "leituras" são ocorrências de falha por família, não sinais contínuos de sensor; os pesos e α/β devem ser recalibrados com dados de sensores.
- Coordenadas e velocidades da rede são aproximadas; os minutos podem ser substituídos por tempos medidos em campo, sem mudar o código.
- O grupo de criticidade (Crítica A/B, Não crítica A/B) não está no dataset; quando disponível, pode entrar como peso no ranking (requisito RN04 da squad).
