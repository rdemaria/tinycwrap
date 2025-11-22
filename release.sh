#!/usr/bin/env bash
set -euo pipefail

NAME=tinycwrap
VER=$(
  python - <<'PY'
import pathlib, tomllib
data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
print(data["project"]["version"])
PY
)

echo "========================================================================"
echo "Preparing release: $NAME v$VER"
echo "========================================================================"

if git status --porcelain | grep -q .; then
  echo "Working tree is dirty. Commit or stash changes before releasing." >&2
  exit 1
fi

if git tag --list "v$VER" | grep -q .; then
  echo "Tag v$VER already exists. Bump the version before releasing." >&2
  exit 1
fi

rm -rf dist/ *.egg-info

echo "Building distribution..."
python -m build

echo "Uploading to pypi..."
twine upload dist/*

echo "Tagging and pushing..."
git tag "v$VER"
git push origin "v$VER"
