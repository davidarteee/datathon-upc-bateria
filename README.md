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

**Leaderboard web (Mode Estàndard):** v1 = **6,43 pts**, 0 infraccions (14/11). El primer en té 6,44: tots els equips bons estan tocant el sostre.

**Novetat (14/11 tarda):** l'agent ara sap quan s'acaba l'episodi (final de mes, perquè el test són els últims 7 dies de cada mes) i no planifica més enllà. Al Mode Difícil val +0,7 punts; a l'Estàndard pot ser el que ens falta si la web no talla la llista de preus a final de setmana.

---

## 3b. 🔥 Mode Difícil (rànquing separat)

> ### 🏁 LLIURAMENT FINAL: `submission_hard_final.zip` — v10 del Dani — **23,32 pts al públic**
>
> El comunicat de l'organització diu que només s'avalua **1 lliurament per equip**, que **només compta el rànquing privat** i que **els premis són només del Mode Difícil**.
>
> **v10 = v6 + tres canvis:** (1) coneix l'estructura del test, que és pública (els últims 7 dies de cada mes): quan l'horitzó passa del final d'una setmana, les hores següents són les de l'inici de la setmana següent; (2) no buida ni força el 50 % al final; (3) Setmana Santa al calendari.
>
> **Comparació en 6 episodis no vistos** ([hard_dani/experiments/robust/v10_compare.py](hard_dani/experiments/robust/v10_compare.py)):
>
> | Versió | No vist | Públic web |
> |---|---|---|
> | v6 | 20,200 | 23,16 |
> | Robusta (tau 20) | 20,207 | 23,15 |
> | **v10** | **20,211** | **23,32** |
> | Robusta + truc de setmanes | 20,221 | — |
>
> Totes empaten en setmanes no vistes (diferències ≤0,02, per sota del llindar de 0,10). Triem la **v10** perquè (a) generalitza igual de bé, (b) la seva millora és **coneixement estructural del test**, que també s'aplica a les setmanes privades (inclou les 12), i no paràmetres ajustats al públic, i (c) és l'única amb aquest mecanisme verificat també al servidor real.
>
> ### ✅ Solució oficial del Mode Difícil: agent del Dani (v6) — **23,16 pts a la web**
>
> **Lliurament:** [`submission_hard.zip`](submission_hard.zip), que conté `agent.py`, `hard_v2_model.py` i `params.json` (SHA-256 `42db7842…20e0c4`). Codi a [submission_hard/](submission_hard/); entrenament a [hard_dani/](hard_dani/) (vegeu-ne el `LLEGEIX-ME.md`).
>
> **Com funciona:**
> 1. **Calendari del campus:** festius (la Mercè, Sant Joan, Nadal...) i tancaments es tracten com a diumenge.
> 2. **Perfils per dia de l'any** (nucli gaussià de 12 dies), amb correccions lineals per horitzó a partir dels residus que l'agent observa en temps real.
> 3. **MPC de 24 h estocàstic** amb 5 escenaris de demanda neta.
> 4. **Pesos del score que s'adapten** al que s'ha vist i **final d'episodi a la càrrega inicial**, perquè l'energia que queda al final compta.
>
> **Validació (setmanes retingudes, episodis continus de 1.008 h, comparat amb la nostra v6):**
>
> | Prova | Màxim | **Dani** | Nostra v6 |
> |---|---|---|---|
> | A setm. 1–6 | 17,30 | **15,64** | 15,17 |
> | A setm. 6–11 | 28,37 | **26,97** | 26,38 |
> | B setm. 1–6 | 16,74 | **15,28** | 15,23 |
> | B setm. 6–11 | 26,26 | **24,89** | 24,53 |
>
> La diferència entre setmanes vistes i no vistes és de només 0,13–0,30 punts, així que **generalitza bé** i no està sobreajustat.
>
> **Reentrenar:** descomprimiu `data/raw/paquet_participants_datathon_upc.zip` a l'arrel del repo, en una carpeta `paquet_participants_datathon_upc/`, i executeu des de `hard_dani/` l'ordre `python -B experiments/dani_hard_v2/train.py final`. Després, des de l'arrel, `python src/make_zip.py`.
>
> ### 🛡️ Versió robusta per al rànquing privat: `submission_hard_robust.zip`
>
> El rànquing final i el jurat fan servir les **6 setmanes privades**, que cap model ha vist. Per això el model que s'entrega s'ha de triar pel que fa en setmanes **no vistes**, no pel score públic. Hem provat 8 variants més regularitzades de la v6 del Dani ([hard_dani/experiments/robust/](hard_dani/experiments/robust/)), validades en 4 episodis de 1.008 h amb setmanes retingudes:
>
> | Variant | No vist | Vist | Diferència |
> |---|---|---|---|
> | v6 Dani | 20,69 | 20,90 | 0,21 |
> | **Perfils més llisos (tau 20) ✅** | **20,72** | 20,83 | **0,11** |
> | Perfils encara més llisos (tau 30) | 20,62 | 20,74 | 0,11 |
> | Ridge 0,1 / Ridge 1 | 20,61 / 20,25 | 20,84 / 20,51 | 0,23 / 0,26 |
> | Menys variables | 20,46 | 20,71 | 0,25 |
> | tau 20 + ridge 0,1 (+ menys variables) | 20,62 (20,41) | 20,77 (20,55) | 0,15 (0,14) |
>
> **tau 20** és la millor en setmanes no vistes i redueix a la meitat la diferència vist/no vist. Simplificar més (ridge alt, menys variables) **empitjora en tot**: això ja és *underfitting*.
>
> **Comitè de models (tau 12+20+30), provat amb 3 grups de setmanes retingudes (6 episodis):** v6 20,200 · robusta 20,207 · comitè 20,204. La regla, fixada abans de veure els resultats, era adoptar el comitè només si guanyava per més de 0,10. No ho fa (−0,003; guanya 3 de 6 episodis), així que **ens quedem la robusta**. Conclusió: el model ja és a prop del seu límit de generalització, i el marge que queda és a les **dades** (setmanes atípiques del test), no a l'ajust.
>
> **Recomanació per a la selecció final (2 lliuraments):** `submission_hard_robust.zip` (la més robusta) + `submission_hard.zip` (la v6 del Dani, 23,16 al públic).
>
> Reentrenar la robusta: `python -B experiments/dani_hard_v2/train.py robust` (des de `hard_dani/`), copiar `params_robust.json` a `submission_hard_robust/params.json` i executar `python src/make_zip.py`.
>
> *Historial:* el que hi ha a continuació documenta la nostra versió anterior (v1→v6, l'última amb 19,22 a la web), que queda al commit `4544bdc`. Els scripts `src/cv_hard.py`, `src/tune_hard.py` i `src/train_forecast.py` són d'aquella versió.

**Què canvia:** bateria **6.000 kWh / ±1.500 kW**, solar **×12** (a migdia sobra energia gratis i neta que, si no es guarda, es perd), i `obs` només dona **3 hores** de preu i CO₂, no 24. A més, `noves_dades_5min` inclou la meteo de l'hora anterior (irradiància, temperatures, vent). Té quota pròpia: 10 intents fins dissabte a les 21 h, 10 fins diumenge a les 8 h i 10 diumenge.

**La nostra solució** ([submission_hard/](submission_hard/)): el mateix MPC-LP, amb tres canvis:
1. **Prediccions** de les 24 h ([src/train_forecast.py](src/train_forecast.py) → `params.json`): preu i CO₂ de l'hora 4 a la 24, i consum i solar de totes. Per a cada variable i cada hora vista endavant es fa una regressió lineal (Ridge) amb aquestes dades: perfil típic (mes × laborable/festiu × hora), valor d'ahir i d'abans-d'ahir, desviació de les últimes hores i, per a la solar, claredat (irradiància observada / típica).
2. **LP amb excedents:** `importació_t ≥ consum_t − solar_t + P_t`, `importació_t ≥ 0`. El model sap que el sol que sobra és gratis i que descarregar més del consum es perd.
3. **Final d'episodi** (final de mes) i un **marge en descarregar**: 25 kW + 40 % de la solar prevista, perquè a migdia la solar fa que el consum net sigui més incert.
4. **LightGBM** (desat com a text dins `params.json`, sense pickle) per a preu i CO₂, en mitjana 50/50 amb la regressió lineal. L'agent precarrega les últimes 72 h de cada mes del train, que són just abans de cada setmana de test.

| Versió (train, per setmanes) | Score | ΔCO₂ | ΔCost | Infr. |
|---|---|---|---|---|
| Oracle (informació perfecta de tota la setmana) | 22,57 | 21,39 % | 25,31 % | 0 |
| MPC amb prediccions perfectes (sense saber el final) | 21,78 | | | 0 |
| hard v1 (prediccions simples) | 19,93 | 18,61 % | 23,00 % | 0 |
| hard v2 (més features + marge 50 kW) | 20,01 | 18,75 % | 22,94 % | 0 |
| hard v3 (+ final d'episodi) | 20,67 | 19,38 % | 23,67 % | 0 |
| **hard v4 (+ LightGBM preu/CO₂ en mitjana amb Ridge, marge segons solar, història precarregada)** | **21,06** | 19,82 % | 23,94 % | **0** |

**Validació honesta** ([src/cv_hard.py](src/cv_hard.py)). Imita el test real: per a cada mes, es guarda l'**última setmana** del train i les prediccions s'entrenen amb la resta. La fem servir per **decidir** (els números de dalt són in-sample i surten optimistes).

| CV (11 setmanes) | Score |
|---|---|
| Oracle | 23,44 |
| MPC + prediccions perfectes (marge 0) | 23,44 |
| hard v3 (Ridge) | 21,35 |
| LightGBM per a tot | 20,96 ❌ (empitjora consum i solar) |
| LightGBM només per a CO₂ | 21,44 |
| Mitjana Ridge/LGB per a preu+CO₂ | 21,48 |
| **+ marge = 25 kW + 0,4·solar prevista (v4)** | **21,61** |

On queda el marge a la CV: preu+CO₂ perfectes **+1,0**, consum+solar perfectes **+0,65**.

### hard v5 — MPC estocàstic (planificar per a diversos futurs)

**Diagnosi** (CV vs oracle): l'agent llençava **24,7 MWh** d'energia: 15,7 MWh de sol que es perdia a migdia perquè la bateria ja era plena, i 9,0 MWh per descarregar més del consum real. L'oracle només en llença 0,5. El problema de fons és que l'MPC determinista es creu la predicció.

**Solució:** a cada hora l'LP planifica alhora per a **13 futurs**: la predicció i 12 escenaris, que són la predicció més errors reals que van cometre els models en el passat a la mateixa hora del dia (camins de 24 h de les 4 variables, presos al mateix instant per conservar la correlació). La decisió d'aquesta hora és **única** i ha de funcionar bé en tots; el pla futur de cada escenari és lliure. Això substitueix els marges "a mà". Si la simulació supera 30 s, l'agent passa automàticament al mode determinista (v4) i mai falla el límit de 60 s.

CV ampliada a **20 setmanes** (les 2 últimes de cada mes):

| CV (20 setmanes) | Score |
|---|---|
| Oracle | 21,72 |
| v4 determinista | 19,84 |
| v4 + biaix solar / marges per quantils | 19,86–19,89 |
| Estocàstic 4 escenaris | 19,92 |
| Estocàstic 8 escenaris | 19,98–20,00 |
| **Estocàstic 12 escenaris, sense marges (v5)** | **20,04** |

Temps: 16 s per a 1.008 h (test públic).

**Resultat web v5: 19,22** (1r: 23,46). Cap diferència de format d'`obs` no explica el gap. La hipòtesi és que les setmanes de test (finals de mes: la Mercè, Nadal, tancament d'agost) són **atípiques**, i les prediccions basades en perfils s'hi equivoquen.

### hard v6 — autocorrecció online
L'agent desa les seves prediccions i corregeix l'error sistemàtic que ha comès en les darreres 24 h (70 %). Test d'estrès amb "campus tancat" (consum ×0,65; oracle 35,59): v5 31,57 → **v6 32,01**. CV normal: **20,07**.

**On és el marge que queda** (punts que guanyaríem amb cada predicció perfecta, sobre la v2): preu + CO₂ **+0,78**, consum + solar **+0,56**, totes **+1,34**. És a dir: **millorar les prediccions és la feina que falta** (Cesc 👀).

**Lliurar:** `python src/make_zip.py` → puja `submission_hard.zip` (agent.py + params.json) a la pestanya *Mode Difícil*.

---

## 4. Pla de treball

- [x] Llegir enunciat, starter kit i dades; muntar el repo
- [x] Pipeline de dades horàries + avaluador local per setmanes
- [x] **Oracle LP amb informació perfecta** → 6,265 (cota superior al train)
- [x] **Agent v1: MPC-LP** amb horitzó 24 h i senyal combinat `w_t` → 6,257, 0 infraccions
- [x] Sensibilitat de `K` i del valor terminal del SOC
- [x] Primer lliurament web (Estàndard): 6,43 pts, 0 infraccions
- [ ] Pujar la v1 + final d'episodi a l'Estàndard i veure si arriba a 6,44
- [x] Mode Difícil: simulador local, oracle, prediccions i agent hard v3 (20,67 al train)
- [ ] Pujar `submission_hard.zip` al Mode Difícil
- [ ] Millorar les prediccions de preu/CO₂ (+0,78 possible) i de consum/solar (+0,56 possible)
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
│   ├── oracle.py      # Cota superior: LP amb informació perfecta (--hard per al Mode Difícil)
│   ├── hard_env.py    # Simulador local del Mode Difícil
│   ├── train_forecast.py  # Entrena les prediccions -> submission_hard/params.json
│   ├── tune_hard.py   # Diagnòstic (quant perdem per cada predicció) i ajust de paràmetres
│   └── make_zip.py    # Crea submission_hard.zip
├── submission_hard/   # Agent del Mode Difícil (agent.py + params.json)
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
