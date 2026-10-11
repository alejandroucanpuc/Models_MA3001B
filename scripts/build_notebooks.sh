#!/usr/bin/env bash
# Script para ejecutar y exportar todos los notebooks de MA3001B a HTML
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NOTEBOOKS_DIR="$REPO_DIR/notebooks"
HTML_DIR="$NOTEBOOKS_DIR/html"
NBCONVERT_BIN="$REPO_DIR/.venv/bin/jupyter"

mkdir -p "$HTML_DIR"

echo "=========================================================="
echo " Ejecutando y exportando Notebooks de MA3001B"
echo " Directorio: $NOTEBOOKS_DIR"
echo " Salida HTML: $HTML_DIR"
echo "=========================================================="

NOTEBOOKS=(
    "N1_RedesSecuenciales_Transformers.ipynb"
    "N2_Estacionariedad_Descomposicion.ipynb"
    "N3_ACF_PACF_CCF_Espectral.ipynb"
    "N4_Atipicos_ARIMA_SARIMA.ipynb"
    "N5_HoltWinters_Benchmark.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
    NB_PATH="$NOTEBOOKS_DIR/$nb"

    if [ -f "$NB_PATH" ]; then
        echo ""
        echo ">>> Procesando: $nb ..."
        "$NBCONVERT_BIN" nbconvert \
            --to html \
            --execute \
            --ExecutePreprocessor.timeout=300 \
            --ExecutePreprocessor.kernel_name=python3 \
            --output-dir "$HTML_DIR" \
            "$NB_PATH"
        echo "    Exportado con éxito en: $HTML_DIR"
    else
        echo "    [AVISO] Archivo no encontrado: $NB_PATH"
    fi
done

echo ""
echo "=========================================================="
echo " Todos los notebooks procesados exitosamente."
echo "=========================================================="
