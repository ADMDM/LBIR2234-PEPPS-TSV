"""Vérifier l'intégrité de tous les fichiers après téléchargement/décompression."""
from pathlib import Path
import hashlib,json,sys

def main():
    root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
    inventory=json.loads((root/'MANIFESTE_SHA256.json').read_text())
    errors=[]
    for r in inventory['files']:
        p=root/r['path'];h=hashlib.sha256()
        if not p.is_file():errors.append(r['path']+': absent');continue
        with p.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):h.update(block)
        if p.stat().st_size!=r['size_bytes'] or h.hexdigest()!=r['sha256']:errors.append(r['path']+': différent')
    print(json.dumps(dict(files=len(inventory['files']),errors=errors,integrity_ok=not errors),ensure_ascii=False,indent=2))
    raise SystemExit(bool(errors))

if __name__=='__main__':main()
