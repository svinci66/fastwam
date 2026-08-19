#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROBOMIMIC_DATA_ROOT:-/home/ubuntu/sj/fastwam/datasets/robomimic_hf/v1.5/can}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

PAIRED_PATH="${ROOT_DIR}/paired/low_dim_v15.hdf5"
MG_PATH="${ROOT_DIR}/mg/low_dim_sparse_v15.hdf5"
PAIRED_BYTES=41692232
MG_BYTES=1098749512
PAIRED_SHA256="5ad06c3591d929eadddc8a7511200a08a58686bfbb22691e966398f0fa005d66"
MG_SHA256="1348033e86dea8bb117c7fe21f044b4239019d690adf07f7ed401253a88161d7"

mkdir -p "$(dirname "${PAIRED_PATH}")" "$(dirname "${MG_PATH}")"

download_and_check() {
    local relative_path="$1"
    local output_path="$2"
    local expected_bytes="$3"
    local expected_sha256="$4"
    wget -c --progress=bar:force:noscroll \
        -O "${output_path}" \
        "${HF_ENDPOINT}/datasets/amandlek/robomimic/resolve/main/${relative_path}"
    local actual_bytes
    actual_bytes="$(stat -c '%s' "${output_path}")"
    if [[ "${actual_bytes}" != "${expected_bytes}" ]]; then
        echo "size mismatch for ${output_path}: expected ${expected_bytes}, got ${actual_bytes}" >&2
        return 1
    fi
    local actual_sha256
    actual_sha256="$(sha256sum "${output_path}" | cut -d ' ' -f 1)"
    if [[ "${actual_sha256}" != "${expected_sha256}" ]]; then
        echo "sha256 mismatch for ${output_path}: expected ${expected_sha256}, got ${actual_sha256}" >&2
        return 1
    fi
}

download_and_check \
    "v1.5/can/paired/low_dim_v15.hdf5" \
    "${PAIRED_PATH}" \
    "${PAIRED_BYTES}" \
    "${PAIRED_SHA256}"
download_and_check \
    "v1.5/can/mg/low_dim_sparse_v15.hdf5" \
    "${MG_PATH}" \
    "${MG_BYTES}" \
    "${MG_SHA256}"

echo "RoboMimic Phase-1 datasets are complete under ${ROOT_DIR}"
