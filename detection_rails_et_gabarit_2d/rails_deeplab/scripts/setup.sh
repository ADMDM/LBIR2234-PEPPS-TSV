#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -e .venv ]; then
    # Persistent application data survives normal cache cleanup and iCloud offloading.
    tsv_project_key=$(python3.12 -c 'import hashlib,pathlib; print(hashlib.sha256(str(pathlib.Path.cwd()).encode()).hexdigest()[:12])')
    case "$(uname -s)" in
        Darwin) tsv_data_root="$HOME/Library/Application Support" ;;
        *) tsv_data_root="${XDG_DATA_HOME:-$HOME/.local/share}" ;;
    esac
    tsv_env_root="$tsv_data_root/tsv-deeplab/model-$tsv_project_key"
    if [ ! -x "$tsv_env_root/bin/python" ]; then
        if [ -e "$tsv_env_root" ]; then
            echo "Environnement incomplet : $tsv_env_root. Conserver ce dossier et choisir un nouvel emplacement." >&2
            exit 1
        fi
        python3.12 -m venv "$tsv_env_root"
    fi
    if [ -L .venv ]; then
        mkdir -p .local
        tsv_old_link=".local/model_lien_interrompu_$(date +%Y%m%d_%H%M%S)_$$"
        mv .venv "$tsv_old_link"
    fi
    ln -s "$tsv_env_root" .venv
fi
if [ -f requirements-lock.txt ]; then
    .venv/bin/python -m pip install -r requirements-lock.txt
else
    .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -m railtsv.cli check
