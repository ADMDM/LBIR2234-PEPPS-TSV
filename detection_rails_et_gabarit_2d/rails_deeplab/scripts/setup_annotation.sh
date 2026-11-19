#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ ! -e .labelme-venv ]; then
    # Persistent application data survives normal cache cleanup and iCloud offloading.
    tsv_project_key=$(python3.12 -c 'import hashlib,pathlib; print(hashlib.sha256(str(pathlib.Path.cwd()).encode()).hexdigest()[:12])')
    case "$(uname -s)" in
        Darwin) tsv_data_root="$HOME/Library/Application Support" ;;
        *) tsv_data_root="${XDG_DATA_HOME:-$HOME/.local/share}" ;;
    esac
    tsv_env_root="$tsv_data_root/tsv-deeplab/labelme-$tsv_project_key"
    if [ ! -x "$tsv_env_root/bin/python" ]; then
        if [ -e "$tsv_env_root" ]; then
            echo "Environnement incomplet : $tsv_env_root. Conserver ce dossier et choisir un nouvel emplacement." >&2
            exit 1
        fi
        python3.12 -m venv "$tsv_env_root"
    fi
    if [ -L .labelme-venv ]; then
        mkdir -p .local
        tsv_old_link=".local/labelme_lien_interrompu_$(date +%Y%m%d_%H%M%S)_$$"
        mv .labelme-venv "$tsv_old_link"
    fi
    ln -s "$tsv_env_root" .labelme-venv
fi
.labelme-venv/bin/python -m pip install -r requirements-labelme-lock.txt
.labelme-venv/bin/python -m pip check
mkdir -p data/annotation_session_v1/images data/annotation_session_v1/json
if [ ! -f data/annotation_session_v1/labels.txt ]; then
    cp configs/annotation_labels.txt data/annotation_session_v1/labels.txt
fi
