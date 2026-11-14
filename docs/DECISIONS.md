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
