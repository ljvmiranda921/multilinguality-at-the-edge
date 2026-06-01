#!/usr/bin/env bash
# Sync the project website (docs/) into the Hugging Face Space submodule (hf-space/).
#   scripts/build_hf_space.sh          # sync docs/ -> hf-space/ (content only)
#   scripts/build_hf_space.sh --push   # sync, commit, and push the submodule to HF
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/docs"
DST="$REPO_ROOT/hf-space"

if [ ! -e "$DST/.git" ]; then
  echo "hf-space submodule not initialised. Run: git submodule update --init hf-space" >&2
  exit 1
fi

# Mirror website content. Protect the submodule's own README.md (the Space card),
# its .gitattributes (LFS rules), and its .git from being touched/deleted.
rsync -a --delete \
  --exclude='.git' --exclude='.gitattributes' --exclude='README.md' --exclude='.DS_Store' \
  "$SRC/" "$DST/"
echo "Synced docs/ -> hf-space/"

if [[ "${1:-}" == "--push" ]]; then
  if ! git lfs version >/dev/null 2>&1; then
    echo "git-lfs is required to push binary assets. Install it first:" >&2
    echo "  brew install git-lfs && git lfs install" >&2
    exit 1
  fi
  cd "$DST"
  git add -A
  git commit -q -m "Sync from docs/ ($(date -u +%Y-%m-%dT%H:%MZ))" || echo "Nothing to commit."
  git push origin main
  echo "Pushed Space. Now commit the submodule pointer in the parent repo:"
  echo "  git -C \"$REPO_ROOT\" add hf-space && git -C \"$REPO_ROOT\" commit -m 'Update hf-space pointer'"
fi
