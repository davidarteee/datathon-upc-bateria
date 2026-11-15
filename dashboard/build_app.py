"""
dashboard/build_app.py
Munta l'app "El Torn de la Bateria" en un sol fitxer HTML (dades, avatars i llibreria a dins).

Ús:
    python -W ignore dashboard/build_data.py   # (si cal) regenera data.json
    python dashboard/build_app.py              # escriu dashboard/index.html i dashboard/artifact.html
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

app = (HERE / "app.html").read_text(encoding="utf-8")
data = (HERE / "data.json").read_text(encoding="utf-8")
avatars = {k: json.loads((HERE / "avatars" / f"{k}.avatar.json").read_text(encoding="utf-8"))
           for k in ("sunee", "cloudee", "cubee")}
bundle = (HERE / "vendor" / "avatar-web.bundle.js").read_text(encoding="utf-8")
assert "</script" not in bundle.lower()

body = (app.replace("/*__AVATAR_BUNDLE__*/", bundle)
           .replace("/*__DATA__*/null", data)
           .replace("/*__AVATARS__*/null", json.dumps(avatars, separators=(",", ":"))))

# artifact.html: contingut sense esquelet (l'Artifact hi afegeix doctype/head/body)
(HERE / "artifact.html").write_text(body, encoding="utf-8")
# index.html: document complet per obrir-lo directament al navegador
full = ('<!doctype html>\n<html lang="ca">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
(HERE / "index.html").write_text(full, encoding="utf-8")
print("index.html", (HERE / "index.html").stat().st_size // 1024, "KB")
