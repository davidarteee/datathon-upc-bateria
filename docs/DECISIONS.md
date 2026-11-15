# Decisions

Registre de decisions tècniques i el seu perquè (servirà per a la presentació al jurat).

## D1 — Optimització abans que Machine Learning (14/11)
**Decisió:** l'eix de la solució és un MPC amb programació lineal sobre l'horitzó de 24 h.
**Per què:** el campus sempre importa més de 500 kW, de manera que l'impacte de la bateria a la xarxa és exactament `P_t`; i preus i CO₂ futurs ja es donen a `obs`. El problema és lineal i quasi determinista: un LP el resol de manera òptima. El ML només aporta en casos límit (consum baix).

## D2 — Zero infraccions per disseny (14/11)
**Decisió:** clamping amb η exacte (`0.9 ** 0.5`) i un petit marge de seguretat.
**Per què:** l'enunciat penalitza −1 punt per infracció (el simulador local només −0,2). Els scores esperats són d'uns pocs punts, així que una sola infracció pot costar més que tota l'optimització.

## D3 — Dades al repo via zip oficial (14/11)
**Decisió:** es versiona només `data/raw/paquet_participants_datathon_upc.zip` (14 MB); els CSV (~90 MB, `inversor.csv` 62 MB) s'extreuen amb `python src/data.py`.
**Per què:** evitar fitxers > 50 MB a GitHub i garantir que tots treballem amb les mateixes dades.

## D4 — v1 = MPC-LP, sense ML (14/11)
**Decisió:** lliurar el MPC-LP de 24 h com a solució base.
**Per què:** arriba a 6,257 sobre un màxim teòric de 6,265 (99,9 %). El marge que queda per a models de predicció és ~0,01 punts al train. A partir d'ara el focus és **robustesa** (que no falli ni faci infraccions al test ocult), validar contra la web i preparar la presentació i el dashboard Tukey.

## D5 — Mode Difícil: mateix MPC-LP + prediccions lineals (14/11)
**Decisió:** reutilitzar el MPC-LP afegint-hi la variable d'importació (per modelar l'excedent solar perdut) i prediccions Ridge per horitzó, desades en JSON.
**Per què:** són ràpides (~5 ms/pas), no depenen de versions de llibreries al servidor (cap pickle) i són explicables al jurat. El diagnòstic amb prediccions perfectes ens diu exactament quant val cada millora de predicció.

## D6 — L'agent sap quan s'acaba l'episodi (14/11)
**Decisió:** no planificar més enllà del final de mes, que és quan acaba cada setmana de test (els últims 7 dies de cada mes).
**Per què:** l'energia que queda a la bateria al final no puntua. Al Mode Difícil (bateria de 6 MWh) val +0,66 punts al train. Si la web executés les setmanes seguides, el cost seria mínim: la bateria es buidaria una mica abans del salt de setmana.

## D7 — Decidim amb validació creuada, no amb el train (14/11)
**Decisió:** cada canvi del Mode Difícil s'accepta només si millora `src/cv_hard.py`, que entrena les prediccions sense l'última setmana de cada mes i avalua l'agent en aquella setmana (com el test real).
**Per què:** in-sample, LightGBM semblava millor a tot arreu, però a la CV empitjora el consum i la solar. Sense CV hauríem pujat una versió pitjor.

## D8 — LightGBM només per a preu/CO2, en mitjana amb Ridge (14/11)
**Per què:** és on la CV mostra millora. La mitjana de dos models diferents redueix l'error. El model es desa com a text (`model_to_string`), per evitar incompatibilitats de versions de pickle al servidor.

## D9 — MPC estocàstic amb escenaris d'errors històrics (14/11)
**Decisió:** l'LP optimitza el cost esperat sobre 13 futurs (predicció + 12 camins d'error reals de la mateixa hora del dia), amb la decisió de l'hora actual comuna a tots.
**Per què:** el determinista es creu la predicció i llença energia (descàrrega excessiva, bateria plena abans del pic solar). Amb escenaris l'agent veu el risc i es cobreix sol, sense marges ajustats a mà. +0,2 a la CV. És un argument fort per al jurat: optimització sota incertesa.

## D10 — Autocorrecció online de les prediccions (14/11)
**Decisió:** l'agent desa les seves prediccions i resta el 70 % de l'error mitjà per horitzó de les darreres 24 h.
**Per què:** el test són els últims 7 dies de cada mes (la Mercè, Nadal, el tancament d'agost...), que són setmanes atípiques i no surten al train. Els models basats en perfils típics s'hi equivoquen de manera sistemàtica. En el test d'estrès amb consum ×0,65 val +0,44, i en setmanes normals no fa mal (+0,03).

## D11 — L'agent del Dani passa a ser la solució del Mode Difícil (15/11)
**Decisió:** `submission_hard/` = agent del Dani v6. El seu codi d'entrenament és a `hard_dani/`.
**Per què:** treu 23,16 a la web (la nostra v6, 19,22) i guanya les 4 validacions amb setmanes retingudes. Gairebé no perd res en setmanes no vistes (0,13–0,30), així que no està sobreajustat al rànquing públic. La clau és el calendari (festius i tancaments) i acabar amb la càrrega inicial.
**Risc conegut:** els festius estan escrits per al curs 2025–26 (és coneixement públic del calendari UPC, no dades del test), i l'agent suposa episodis web de 1.008 o 2.016 h.

## D12 — Versió robusta (tau 20) per al rànquing privat (15/11)
**Decisió:** `submission_hard_robust/` = v6 del Dani amb perfils per dia de l'any més llisos (tau 20).
**Per què:** el rànquing privat són setmanes que cap model ha vist. A la validació amb setmanes retingudes, tau 20 té el millor score en setmanes no vistes (20,72) i la meitat de diferència vist/no vist (0,11 contra 0,21). Les variants més simples (ridge alt, menys variables) són pitjors en tot. Per a la selecció final de 2 lliuraments: la robusta + la v6 del Dani.
