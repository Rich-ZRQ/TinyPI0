#!/usr/bin/env bash
set -euo pipefail

# Do not inherit the SSH client's reverse/local proxy. The mirror is reached
# directly from this server, so a dropped SSH connection cannot interrupt the
# network path used by huggingface_hub.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY
unset http_proxy https_proxy all_proxy

export HF_HOME="${TINYPI0_HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# Avoid redirecting large files to the Xet backend, which may not be reachable
# on the same route as the configured endpoint.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ $# -eq 0 ]]; then
    cat >&2 <<'EOF'
Usage: scripts/hf_download_server.sh REPO_ID [FILES ...] [hf download options]

Examples:
  scripts/hf_download_server.sh google/paligemma2-3b-pt-224
  scripts/hf_download_server.sh owner/dataset --repo-type dataset

Set HF_ENDPOINT before running to override https://hf-mirror.com.
EOF
    exit 2
fi

echo "Hugging Face endpoint: ${HF_ENDPOINT}" >&2
exec hf download "$@"
