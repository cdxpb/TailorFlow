# source env.sh   (activates the conda environment created from environment.yml)
if command -v micromamba >/dev/null 2>&1 || [ -n "$MAMBA_EXE" ]; then
    eval "$("${MAMBA_EXE:-micromamba}" shell hook -s bash)" && micromamba activate tailorflow
else
    eval "$(conda shell.bash hook)" && conda activate tailorflow
fi
export CUDA_HOME="$CONDA_PREFIX"
export TORCH_EXTENSIONS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.torch_extensions"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
