#!/usr/bin/env bash
# Sube el código (commit actual) como dataset privado y lanza el kernel de entrenamiento.
# Uso: kaggle/subir.sh [crear]   (usar "crear" la primera vez)
set -euo pipefail
cd "$(dirname "$0")/.."
K="uvx --python 3.12 --with six kaggle"
stage=$(mktemp -d)
git archive HEAD | tar -x -C "$stage"
rm -rf "$stage/kaggle" "$stage/docs" "$stage/notebooks"
cat > "$stage/dataset-metadata.json" <<J
{"title": "darkforest-code", "id": "biancacaldern/darkforest-code", "licenses": [{"name": "other"}]}
J
if [ "${1:-}" = "crear" ]; then
  $K datasets create -p "$stage" --dir-mode zip
else
  $K datasets version -p "$stage" -m "código $(git rev-parse --short HEAD)" --dir-mode zip
fi
$K kernels push -p kaggle
