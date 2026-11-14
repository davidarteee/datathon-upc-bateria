# 🔋 Carrega la Bateria! — Repte UPC Sostenible · Datathon FME 2026

**Equip:** Oriol · Cesc · Dani · David
**Repte:** dissenyar l'algorisme de control (`BatteryAgent`) d'una bateria industrial de **2.000 kWh / ±500 kW** al Campus Nord de la UPC, decidint **hora a hora** quanta potència carregar o descarregar per reduir les **emissions de CO₂ (70 %)** i la **factura elèctrica (30 %)** respecte al campus sense bateria.
**Plataforma oficial:** https://repteupc.datathon.cat/
**Side challenge:** Premi Tukey — un dashboard que ajudi a prendre decisions sobre el repte (vegeu [enunciat/SideChallenge_Tukey.pdf](enunciat/SideChallenge_Tukey.pdf)).

---

## 1. El problema en 1 minut

A cada hora `t` l'agent rep `obs` (hora, dia, SOC, preu i CO₂ actuals, **preus i CO₂ de les properes 24 h**, i consum/solar de l'hora anterior) i retorna una consigna `P_t` en kW:

| `P_t` | Acció |
|---|---|
| `> 0` | Carregar (fins a +500 kW) |
| `< 0` | Descarregar cap al campus (fins a −500 kW) |
| `= 0` | Repòs |

**Física de la bateria** ([battery_spec.json](starter_kit/battery_spec.json)): capacitat 2.000 kWh, SOC inicial 50 %, rendiment d'anada i tornada 90 % (η_c = η_d = √0,9 ≈ 0,9487).

```
Càrrega:     SOC_{t+1} = SOC_t + P_t · η_c
Descàrrega:  SOC_{t+1} = SOC_t + P_t / η_d
Xarxa:       P_xarxa,t = consum_t − solar_t + P_t      (l'excedent exportat no es paga: preu 0)
```

**Puntuació oficial:**

```
ΔCO₂(%)  = (CO₂_base − CO₂_agent) / CO₂_base · 100
ΔCost(%) = (Cost_base − Cost_agent) / Cost_base · 100
Score    = max(0, 0,70·ΔCO₂ + 0,30·ΔCost − 1,00·Infraccions)
```

Infracció = consigna fora de ±500 kW o que portaria el SOC fora de [0, 2.000] kWh.

**Avaluació:** test ocult = els **últims 7 dies de cada mes** (~12 setmanes, ~1.700–2.000 h), simulació seqüencial sense *data leakage*, rànquing 50 % públic / 50 % privat, + valoració del jurat (presentació i raonament).

**Restriccions del lliurament:** `agent.py` (o `submission.zip` amb `.pkl`/`.json`), ≤ 25 MB, **≤ 60 s** per simulació, **10 lliuraments/dia**, només llibreries permeses (numpy, pandas, scipy, scikit-learn, joblib, xgboost, lightgbm…). Prohibit `os`, `sys`, `subprocess`, xarxa, `eval/exec`… (vegeu [check_submission.py](starter_kit/check_submission.py)).

---

## 2. Troballes clau (el que ens farà guanyar)

Anàlisi inicial sobre les 5.993 h de train (set 2025 – ago 2026):

1. **El campus sempre importa de la xarxa.** El consum mitjà és ~1.120 kW i la solar màxima només ~144 kW; mai hi ha excedent solar i el consum gairebé mai baixa de 500 kW (16 h de 5.993). Per tant, **l'efecte de la bateria a la xarxa és exactament `P_t`**: el problema és pràcticament **lineal i determinista**.
2. **Preus i CO₂ de les properes 24 h es donen exactes a `obs`.** No cal predir-los. El consum/solar quasi no afecten la decisió (pel punt 1) → la clau no és el ML sinó l'**optimització**.
3. **Es pot fusionar l'objectiu en un sol senyal per kWh:**
   `w_t = 0,70 · co2_t / CO₂_base + 0,30 · preu_t / Cost_base`
   i minimitzar `Σ w_t · P_t` → un **LP de 24 variables per hora (MPC)**, resolt amb `scipy.optimize.linprog` en mil·lisegons (cap problema amb el límit de 60 s).
4. **Perfil diari** (mitjanes per hora): el CO₂ és **mínim a migdia (11–15 h, ~0,052)** i màxim al vespre/matinada (~0,11); el preu és mínim a les 14–16 h (~0,08 €/kWh) i màxim a les 19–21 h (~0,23 €/kWh). Correlació preu–CO₂ = 0,58 → els dos objectius en general van alineats: **carregar a migdia, descarregar al vespre**. Rendiment 90 % → només val la pena un cicle si `w_descàrrega · η > w_càrrega / η`.
5. **⚠️ La plantilla oficial ja fa infraccions** (8 a la simulació setmanal, 46 a la completa) per arrodonir η a 0,9487. Amb −1 punt per infracció això és letal. **Clamping estricte amb marge** obligatori.
6. **⚠️ Discrepància:** l'`environment.py` local penalitza 0,2 punts/infracció, l'enunciat diu **1,00**. Assumim el pitjor cas.
7. **Final de període:** cada setmana de test comença al 50 % de SOC; l'energia que queda a la bateria al final no puntua → cal decidir el valor terminal del SOC (no és obvi si el simulador reinicia per setmana o no; cal verificar-ho amb el dry-run web).

---

## 3. Resultats (log viu)

Avaluat amb `python src/evaluate.py <agent.py>` sobre el train, en trossos d'1 setmana amb SOC inicial 50 % (simula el test). Detall a [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

| Versió | Score (sense penal.) | ΔCO₂ % | ΔCost % | Infraccions | Score oficial (−1/infr.) |
|---|---|---|---|---|---|
| Plantilla oficial (heurística) | 1,76 | 1,33 | 2,77 | 8 | **0** |
| **v1 MPC-LP** ([submission/agent.py](submission/agent.py)) | **6,257** | 5,84 | 7,24 | **0** | **6,257** |
| _Oracle (informació perfecta, cota superior)_ | _6,265_ | _5,85_ | _7,24_ | _0_ | — |

**La v1 assoleix el 99,9 % del màxim teòric** amb 0 infraccions i ~4 ms per hora (~8 s per a tot el test; el límit és 60 s).

**Sensibilitat (v1):** el valor terminal del SOC no canvia res (l'horitzó de 24 h ja captura el cicle diari). El pes del preu `K` és l'únic paràmetre que importa, i és robust: si el multipliquem o el dividim per 2, el score només baixa ~0,07; amb `K = 0` (optimitzar només CO₂) cau a 5,74.

### Com funciona la v1

1. Cada hora es combina CO₂ i preu en un pes per kWh: `w_t = co2_t + K·preu_t`, amb `K = (0,3/0,7)·(CO₂_base/Cost_base) ≈ 0,252`. Aquest `K` s'estima al train (CO₂_base/Cost_base = 0,587 kg/€).
2. Es resol un LP sobre les properes 24 h (variables: càrrega i descàrrega de cada hora; restriccions: SOC a [0, 2.000], potència ≤ 500 kW, rendiment de 90 %) que minimitza `Σ w_t·P_t`.
3. S'aplica només la primera hora (MPC) i es torna a planificar a l'hora següent amb la informació nova.
4. La descàrrega es limita al consum net de l'última hora (exportar no es paga).
5. Hi ha un clamping final amb el η exacte i un marge de 1e-6, de manera que és impossible fer infraccions.

---

## 4. Pla de treball

- [x] Llegir enunciat, starter kit i dades; muntar el repo
- [x] Pipeline de dades horàries + avaluador local per setmanes
- [x] **Oracle LP amb informació perfecta** → 6,265 (cota superior al train)
- [x] **Agent v1: MPC-LP** amb horitzó 24 h i senyal combinat `w_t` → 6,257, 0 infraccions
- [x] Sensibilitat de `K` i del valor terminal del SOC
- [ ] Validació web (dry-run) i primer lliurament al leaderboard: comprovar que el score web quadra amb el local
- [ ] Verificar a la web si el test reinicia la bateria per setmana o és una sola simulació
- [ ] Robustesa: casos límit (consum < 500 kW, buits de dades, llistes de 24 h incompletes)
- [ ] Dashboard Premi Tukey (decisions hora a hora, comparativa vs base, sensibilitat)
- [ ] Presentació final (repartiment entre els 4)

---

## 5. Estructura del repo

```text
├── enunciat/          # Material oficial: presentació, ENUNCIAT.md, guia de variables, side challenge Tukey
├── starter_kit/       # Starter kit oficial SENSE modificar (environment.py, check_submission.py, agent.py plantilla…)
├── data/
│   ├── raw/           # Zip oficial del paquet de participants (font de veritat de les dades)
│   ├── train/         # (ignorat per git) CSVs extrets automàticament
│   └── processed/     # (ignorat per git) hourly.csv generat
├── src/
│   ├── data.py        # Extreu el zip i construeix la taula horària (mateix preprocessament que el tutorial)
│   ├── evaluate.py    # Avaluació local: tot el train + per setmanes (com el test)
│   └── oracle.py      # Cota superior: LP amb informació perfecta de tota la setmana
├── submission/        # El NOSTRE agent.py (i .pkl/.json si en calen) → el que es puja a la web
├── notebooks/         # EDA i experiments
└── docs/              # EXPERIMENTS.md (resultats) i DECISIONS.md (per què fem cada cosa)
```

## 6. Com començar

```bash
git clone <url-del-repo> && cd datathon-upc-bateria
pip install -r requirements.txt
python src/data.py                                   # extreu i prepara les dades (1 cop)
python src/evaluate.py submission/agent.py           # puntua el nostre agent
python starter_kit/check_submission.py submission/agent.py   # validador oficial abans de pujar
```

---

*Bases de la Datathon: https://www.datathon.cat/condicions*
