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
