#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCHIVE="${1:-}"
REF_DIR="$ROOT/reference/v2_mla_poc"

mkdir -p "$ROOT"/{reference,vendor,src/v4_kv_quant,tests,tools,configs,results,artifacts/env,docs}

if [[ -n "$ARCHIVE" ]]; then
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: archive not found: $ARCHIVE" >&2
    exit 1
  fi
  if [[ -e "$REF_DIR" ]]; then
    echo "ERROR: reference directory already exists: $REF_DIR" >&2
    echo "Remove it manually only if you are certain it is safe." >&2
    exit 1
  fi
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  unzip -q "$ARCHIVE" -d "$TMP"
  TOP_COUNT=$(find "$TMP" -mindepth 1 -maxdepth 1 | wc -l)
  if [[ "$TOP_COUNT" -eq 1 && -d "$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1)" ]]; then
    SRC="$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1)"
  else
    SRC="$TMP"
  fi
  cp -a "$SRC" "$REF_DIR"
  chmod -R a-w "$REF_DIR"
  echo "Reference repository unpacked read-only at: $REF_DIR"
else
  echo "No archive supplied. Workspace directories created."
  echo "Usage: bash scripts/bootstrap_workspace.sh /path/to/reference.zip"
fi

cat <<MSG

Workspace ready: $ROOT

Next:
  bash scripts/capture_environment.sh
  git init
  git add .
  git commit -m "Initialize DeepSeek V4 KV quantization project"
  claude
MSG
