#!/usr/bin/env bash
# Installs the HCUP leak guard as this repo's pre-commit hook.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .git ]; then
  echo "Not a git repo yet. Run 'git init' first, then re-run this script."
  exit 1
fi

cp tools/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
echo "Installed .git/hooks/pre-commit"
echo "Test it with: git commit --allow-empty -m test"
