# La MasIA · Mode Difícil · entrenament de l'agent

Aquest paquet conté el codi que genera `params.json`, el fitxer amb el model entrenat que va dins de cada ZIP enviat al portal.
No conté dades: els CSV són els del paquet oficial de participants.

## Estructura

| Fitxer | Què fa |
|---|---|
| `experiments/dani_hard_v2/train.py` | Entrena l'agent i escriu `params.json` (`final`) o valida amb setmanes retingudes (`validate`). |
| `experiments/dani_hard_v2/model.py` | L'agent: previsió causal i MPC de 24 h. Al ZIP del portal es diu `hard_v2_model.py`. |
| `experiments/dani_hard_v2/agent.py` | Punt d'entrada `BatteryAgent` que llegeix `params.json`. |
| `experiments/dani_hard_v2/package.py` | Empaqueta el ZIP, el passa pel verificador oficial i en fa un episodi aïllat. |
| `experiments/dani_hard_v2/params.json` | Paràmetres de referència de la v6 (SHA-256 `42db78427d642182e0d59bf604e63f094b1f08b53a6b10e3eb0be9e16b20e0c4`). |
| `experiments/oriol_hard_mode/lab.py` | Motor oficial, preprocessament de les dades i solar ×12 (de l'Oriol, sense canvis). |
| `experiments/dani_hard_score_estimation/estimate.py` | Blocs setmanals i sostre amb previsió perfecta, per validar. |
| `scripts/evaluate.py` | Lectura i agregació horària dels CSV oficials. |

La resta de fitxers d'`oriol_hard_mode` hi són perquè `train.py` i `estimate.py` els importen.

## Com es fa servir

1. Descomprimeix el paquet oficial de participants al costat d'aquesta carpeta, amb aquest nom exacte:

```
carpeta-qualsevol/
├── la-masia-hard-train/                  (aquesta carpeta)
└── paquet_participants_datathon_upc/     (amb data/train/*.csv i starter_kit/)
```

2. Instal·la les dependències (Python 3.11 o més nou):

```bash
python -m pip install -r requirements.txt
```

3. Des de dins de `la-masia-hard-train/`:

```bash
python -B experiments/dani_hard_v2/train.py final
```

Escriu `experiments/dani_hard_v2/params.json`. Amb les mateixes versions de numpy, pandas i scipy surt idèntic al de referència.

Opcional:

```bash
python -B experiments/dani_hard_v2/train.py validate
```

```bash
python -B experiments/dani_hard_v2/package.py
```

`validate` entrena retirant setmanes senceres i compara amb la versió congelada de l'Oriol. `package.py` genera `hard_v6_submission.zip` i el comprova.

## Què aprèn `train.py final`

- Perfils de demanda per dia de l'any, hora i tipus de dia (laborable, dissabte o activitat reduïda, diumenge/festiu/tancat), amb nucli gaussià de 12 dies.
- Perfils de solar mitjana i de cel clar (quantil 90 %).
- Perfils de preu i CO₂ per dia de l'any, hora i laborable/festiu.
- Correccions lineals per horitzó (0–23 h) a partir dels residus que l'agent veu en temps real.
- Desviació de l'error de demanda neta per horitzó i hora, per al pla amb 5 escenaris.
- Configuració de la v6: pla estocàstic, pesos adaptatius i cost de l'energia que falti al final per tornar a la càrrega inicial.
