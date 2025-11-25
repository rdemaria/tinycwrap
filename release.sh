#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [major|minor|bug]" >&2
  exit 1
}

BUMP=${1:-}
case "$BUMP" in
  major|minor|bug|patch) ;;
  *) usage ;;
esac
[ "$BUMP" = "patch" ] && BUMP="bug"

NAME=tinycwrap

if git status --porcelain | grep -q .; then
  echo "Working tree is dirty. Commit or stash changes before releasing." >&2
  exit 1
fi

CURRENT_VER=$(
  python - <<'PY'
import pathlib, tomllib
data = tomllib.loads(pathlib.Path("pyproject.toml").read_text())
print(data["project"]["version"])
PY
)

NEW_VER=$(
  python - "$BUMP" <<'PY'
import pathlib, tomllib, re, sys

bump = sys.argv[1]
pyproj = pathlib.Path("pyproject.toml")
data = tomllib.loads(pyproj.read_text())
ver = data["project"]["version"]
major, minor, patch = map(int, ver.split("."))
if bump == "major":
    major += 1
    minor = 0
    patch = 0
elif bump == "minor":
    minor += 1
    patch = 0
else:
    patch += 1
new_ver = f"{major}.{minor}.{patch}"

text = pyproj.read_text()
text = re.sub(r'(?m)^version\s*=\\s*\"[^\"]+\"', f'version = "{new_ver}"', text)
pyproj.write_text(text)

init_py = pathlib.Path("tinycwrap/__init__.py")
init_text = init_py.read_text()
init_text = re.sub(r'(__version__\\s*=\\s*\")[^\"]+(\")', rf'\\1{new_ver}\\2', init_text, count=1)
init_py.write_text(init_text)

print(new_ver)
PY
)

echo "========================================================================"
echo "Preparing release: $NAME $CURRENT_VER -> $NEW_VER"
echo "========================================================================"

if git tag --list "v$NEW_VER" | grep -q .; then
  echo "Tag v$NEW_VER already exists. Aborting." >&2
  exit 1
fi

rm -rf dist/ *.egg-info

git add pyproject.toml tinycwrap/__init__.py
git commit -m "Release v$NEW_VER"

echo "Building distribution..."
python -m build

echo "Uploading to pypi..."
twine upload dist/*

echo "Tagging and pushing..."
git tag "v$NEW_VER"
git push origin HEAD
git push origin "v$NEW_VER"
