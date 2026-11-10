#!/usr/bin/env bash
# Sube el código del árbol de trabajo como dataset privado y lanza el kernel de entrenamiento.
# Uso: kaggle/subir.sh [crear]   (usar "crear" la primera vez)
# Variables: KAGGLE_OWNER (usuario de Kaggle; por defecto biancacaldern) y DF_ACCELERATOR
# (por defecto NvidiaTeslaT4, que en Kaggle es GPU T4 x2). Las credenciales las toma el CLI de
# ~/.kaggle o de KAGGLE_USERNAME y KAGGLE_API_TOKEN; este script no las lee ni las imprime.
set -euo pipefail
cd "$(dirname "$0")/.."
OWNER="${KAGGLE_OWNER:-biancacaldern}"
K="uvx --python 3.12 --with six kaggle"
stage=$(mktemp -d)
kernel=$(mktemp -d)
git ls-files -co --exclude-standard -z | grep -zv '^\(kaggle/\|docs/\|notebooks/\|\.\)' |
  while IFS= read -r -d '' f; do [ -f "$f" ] && install -D "$f" "$stage/$f"; done
cat > "$stage/dataset-metadata.json" <<J
{"title": "darkforest-code", "id": "$OWNER/darkforest-code", "licenses": [{"name": "other"}]}
J
sed "s#biancacaldern#$OWNER#g" kaggle/kernel-metadata.json > "$kernel/kernel-metadata.json"
cp kaggle/run.py "$kernel/run.py"
if [ "${1:-}" = "crear" ]; then
  $K datasets create -p "$stage" --dir-mode zip
else
  $K datasets version -p "$stage" -m "código $(git rev-parse --short HEAD) + cambios locales" --dir-mode zip
fi
$K kernels push -p "$kernel" --accelerator "${DF_ACCELERATOR:-NvidiaTeslaT4}"
