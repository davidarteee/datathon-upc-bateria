# Experiments

Cada experiment: data/hora, qui, què s'ha provat, resultat de `python src/evaluate.py` (mode **per setmanes**) i conclusió.
Si es puja al leaderboard, apunteu-hi també el score públic.

| # | Data | Qui | Agent | Score s/penal. | ΔCO₂ % | ΔCost % | Infr. | Leaderboard | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 14/11 | David | `starter_kit/agent.py` (plantilla) | 1,758 | 1,326 | 2,766 | 8 | — | Les infraccions venen d'arrodonir η a 0,9487 al clamping. Amb −1/infr. el score oficial seria 0. |
| 1 | 14/11 | David | `src/oracle.py` (informació perfecta) | 6,265 | 5,847 | 7,240 | 0 | — | Cota superior al train. Cal escalar `w` a ~1 o HiGHS retorna solucions dolentes (pes ~1e-7). |
| 2 | 14/11 | David | `submission/agent.py` v1 MPC-LP 24 h, K=0,252 | 6,257 | 5,837 | 7,239 | 0 | pendent | 99,9 % de l'oracle. ~4 ms/pas. |
| 2b | 14/11 | David | v1 amb K=0 / K·0,5 / K·2 | 5,743 / 6,195 / 6,182 | | | 0 | — | K estimat és l'òptim; és robust a errors de ×2. |
| 2c | 14/11 | David | v1 amb valor terminal q=0,3 / 0,7 / 0 | 6,257 | | | 0 | — | El valor terminal no influeix. |
| 3 | 14/11 | David | v1 al leaderboard web (Estàndard) | — | 5,97 | 7,50 | 0 | **6,43** | 1r = 6,44. Molts equips amb exactament els nostres números → tothom al sostre. |
| 4 | 14/11 | David | `src/oracle.py --hard` | 22,566 | 21,389 | 25,312 | 0 | — | Cota superior Mode Difícil. K_hard = 0,2617. |
| 5 | 14/11 | David | hard v1: MPC + Ridge (perfil, ahir, darrera desviació) | 19,930 | 18,61 | 23,00 | 0 | — | 88 % de l'oracle. |
| 6 | 14/11 | David | hard v2: + ahir-2, desviació mitjana 24 h, marge descàrrega 50 kW | 20,007 | 18,75 | 22,94 | 0 | — | Marge 150 kW: 19,25 (massa). Valor terminal i H=36/72: cap efecte. |
| 6b | 14/11 | David | diag v2: prediccions perfectes de preu+CO2 / consum / solar / tot | 20,79 / 20,26 / 20,28 / 21,35 | | | 0 | — | Amb tot perfecte i H=72 encara 21,78 < 22,57 → el que falta és saber on acaba la setmana. |
| 7 | 14/11 | David | **hard v3: + final d'episodi (final de mes)** | **20,666** | 19,38 | 23,67 | 0 | pendent | +0,66. Estàndard local sense canvis (6,257), però a la web pot ajudar. |
| 8 | 14/11 | David | **CV honesta** (`src/cv_hard.py`, 11 setmanes = última setmana de cada bloc) | oracle 23,44 · v3 21,35 | | | 0 | — | A partir d'ara decidim amb la CV, no in-sample. |
| 9 | 14/11 | David | LightGBM (features = `BatteryAgent._feats`) per a tot / preu / CO2 / consum / solar | 20,96 / 21,42 / 21,44 / 21,18 / 21,12 (CV) | | | 0 | — | Només ajuda en preu i CO2. |
| 10 | 14/11 | David | Mitjana 50/50 Ridge+LGB (preu, CO2) | 21,48 (CV) | | | 0 | — | lgb_w 0,7: 21,63 (soroll); ens quedem amb 0,5. |
| 11 | 14/11 | David | Marge descàrrega 25 kW + 0,4·solar prevista | **21,61 (CV)** · 21,06 (42 setm.) | 19,82 | 23,94 | 0 | pendent | Marge 0: 21,31; 100 fix: 21,16. K 0,20/0,33: ±0,03 (indiferent). |
| 12 | 14/11 | David | Diagnosi CV vs oracle | — | | | | — | Llencem 24,7 MWh (15,7 de sol perdut a migdia + 9,0 per descarregar més del consum); l'oracle en llença 0,5. |
| 13 | 14/11 | David | CV ampliada a 20 setmanes; v4 / biaix solar 1,3 / quantil marge 0,9 | 19,84 / 19,89 / 19,86 | | | 0 | — | Oracle CV20 = 21,72. Retocs del determinista ≈ soroll. |
| 14 | 14/11 | David | **v5 MPC estocàstic** 4 / 8 / 12 escenaris d'error històric | 19,92 / 20,00 / **20,04** | 18,69 | 23,20 | 0 | pendent | 16 s/1.008 h. Fallback determinista si >30 s. |
