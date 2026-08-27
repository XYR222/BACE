#!/usr/bin/env bash
# Historical RWTH-only migration script. Review every absolute path before use;
# this is not the portable installation entry documented under docs/02.
set -euo pipefail

SOURCE_ENV=/hpcwork/xsz96350/fu_project/verl-agent
TARGET_ROOT=/hpcwork/rwth2089/xsz96350/work-BACE
FINAL_ENV=${TARGET_ROOT}/verl-agent
BACKUP_ROOT=${TARGET_ROOT}/environment-backups
MIGRATION_ID=${MIGRATION_ID:-${SLURM_JOB_ID:-manual_$(date -u +%Y%m%dT%H%M%SZ)}}
STAGING_ENV=${TARGET_ROOT}/.verl-agent.staging-${MIGRATION_ID}
ARCHIVE=${BACKUP_ROOT}/verl-agent-${MIGRATION_ID}.tar.zst
ARCHIVE_PARTIAL=${ARCHIVE}.partial
MANIFEST=${BACKUP_ROOT}/verl-agent-${MIGRATION_ID}.manifest.txt
VALIDATION=${BACKUP_ROOT}/verl-agent-${MIGRATION_ID}.validation.txt

[[ -d "${SOURCE_ENV}" && ! -L "${SOURCE_ENV}" ]] || {
    printf 'Source environment is missing or already a symlink: %s\n' "${SOURCE_ENV}" >&2
    exit 2
}
[[ ! -e "${FINAL_ENV}" && ! -L "${FINAL_ENV}" ]] || {
    printf 'Final target already exists: %s\n' "${FINAL_ENV}" >&2
    exit 2
}
[[ ! -e "${STAGING_ENV}" && ! -L "${STAGING_ENV}" ]] || {
    printf 'Staging target already exists: %s\n' "${STAGING_ENV}" >&2
    exit 2
}
[[ ! -e "${ARCHIVE}" && ! -e "${ARCHIVE_PARTIAL}" ]] || {
    printf 'Backup archive already exists for migration id %s\n' "${MIGRATION_ID}" >&2
    exit 2
}

mkdir -p "${BACKUP_ROOT}" "${STAGING_ENV}"

source_entries=$(find "${SOURCE_ENV}" -xdev -printf . | wc -c)
source_bytes=$(du -sx -B1 "${SOURCE_ENV}" | awk '{print $1}')
source_path_digest=$(
    find "${SOURCE_ENV}" -xdev -printf '%P\0' | LC_ALL=C sort -z | sha256sum | awk '{print $1}'
)

{
    printf 'migration_id=%s\n' "${MIGRATION_ID}"
    printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'source_env=%s\n' "${SOURCE_ENV}"
    printf 'target_env=%s\n' "${FINAL_ENV}"
    printf 'source_entries=%s\n' "${source_entries}"
    printf 'source_bytes=%s\n' "${source_bytes}"
    printf 'source_path_digest=%s\n' "${source_path_digest}"
    "${SOURCE_ENV}/bin/python" -V
    "${SOURCE_ENV}/bin/python" -m pip freeze
} > "${MANIFEST}"

tar --acls --xattrs -C "${SOURCE_ENV}" -cf - . \
    | zstd -T"${SLURM_CPUS_PER_TASK:-8}" -3 -o "${ARCHIVE_PARTIAL}"
zstd -t "${ARCHIVE_PARTIAL}"
mv "${ARCHIVE_PARTIAL}" "${ARCHIVE}"
sha256sum "${ARCHIVE}" >> "${MANIFEST}"

zstd -dc "${ARCHIVE}" | tar --acls --xattrs -C "${STAGING_ENV}" -xf -

staging_entries=$(find "${STAGING_ENV}" -xdev -printf . | wc -c)
staging_bytes=$(du -sx -B1 "${STAGING_ENV}" | awk '{print $1}')
staging_path_digest=$(
    find "${STAGING_ENV}" -xdev -printf '%P\0' | LC_ALL=C sort -z | sha256sum | awk '{print $1}'
)

{
    printf 'staging_env=%s\n' "${STAGING_ENV}"
    printf 'staging_entries=%s\n' "${staging_entries}"
    printf 'staging_bytes=%s\n' "${staging_bytes}"
    printf 'staging_path_digest=%s\n' "${staging_path_digest}"
    [[ "${source_entries}" == "${staging_entries}" ]]
    [[ "${source_path_digest}" == "${staging_path_digest}" ]]
    "${STAGING_ENV}/bin/python" - <<'PY'
import importlib.metadata
import pathlib
import sys

expected = pathlib.Path(sys.argv[0]).resolve().parent.parent
actual = pathlib.Path(sys.prefix).resolve()
print(f"python={sys.version}")
print(f"executable={sys.executable}")
print(f"prefix={sys.prefix}")
for package in (
    "torch",
    "vllm",
    "xformers",
    "ray",
    "transformers",
    "flash_attn",
    "alfworld",
):
    print(f"{package}={importlib.metadata.version(package)}")
import torch
import vllm
import xformers
import ray
import transformers
import flash_attn
import alfworld
print(f"torch_cuda={torch.version.cuda}")
print("core_imports=ok")
PY
    "${STAGING_ENV}/bin/python" -m pip check
} > "${VALIDATION}" 2>&1

mv "${STAGING_ENV}" "${FINAL_ENV}"
{
    printf 'final_env=%s\n' "${FINAL_ENV}"
    printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'status=ready_for_gpu_validation\n'
} >> "${VALIDATION}"

printf 'Backup: %s\n' "${ARCHIVE}"
printf 'Manifest: %s\n' "${MANIFEST}"
printf 'Target environment: %s\n' "${FINAL_ENV}"
printf 'Validation: %s\n' "${VALIDATION}"
