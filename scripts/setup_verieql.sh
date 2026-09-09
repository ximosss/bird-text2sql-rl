#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${VERIEQL_PATH:-$project_root/third_party/VeriEQL}"
revision="493cbb81000205e33b0623cfd1c39106fa035fae"
repository="https://github.com/VeriEQL/VeriEQL.git"

if [[ -f "$target/environment.py" ]]; then
  observed="$(git -C "$target" rev-parse HEAD 2>/dev/null || true)"
  if [[ "$observed" == "$revision" ]]; then
    printf 'VeriEQL is ready: %s (%s)\n' "$target" "$revision"
    exit 0
  fi
  printf 'Existing VeriEQL checkout is not at the pinned revision: %s (%s)\n' \
    "$target" "${observed:-unknown}" >&2
  exit 1
fi
if [[ -e "$target" ]]; then
  printf 'Refusing to overwrite existing path: %s\n' "$target" >&2
  exit 1
fi

proxy_configured=false
for variable in HTTPS_PROXY https_proxy HTTP_PROXY http_proxy ALL_PROXY all_proxy; do
  if [[ -n "${!variable:-}" ]]; then
    proxy_configured=true
    break
  fi
done
if [[ "$proxy_configured" == false ]] && ! git config --get http.proxy >/dev/null 2>&1; then
  printf 'No local proxy was detected. Configure HTTPS_PROXY/HTTP_PROXY or git http.proxy first.\n' >&2
  exit 1
fi

mkdir -p "$(dirname "$target")"
git clone --filter=blob:none --no-checkout "$repository" "$target"
git -C "$target" checkout --detach "$revision"
printf 'VeriEQL is ready: %s (%s)\n' "$target" "$revision"
