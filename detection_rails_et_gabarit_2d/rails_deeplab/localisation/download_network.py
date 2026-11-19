"""Fetch the public, pinned African rail network; never transmit user coordinates."""
from pathlib import Path
import hashlib
import json
import urllib.request

RECORD_URL = "https://zenodo.org/api/records/17861120"
EXPECTED = {"africa_railways_network.gpkg": "f243222394e1db6a5f719f34570a92a9",
            "METADATA.xlsx": "4b8ce35d0336256aa6bf861a3959eaaf"}


def main():
    out = Path(__file__).resolve().parent / "donnees"
    out.mkdir(exist_ok=True)
    with urllib.request.urlopen(RECORD_URL, timeout=60) as response:
        record = json.load(response)
    for file in record["files"]:
        if file["key"] not in EXPECTED:
            continue
        target = out / file["key"]
        if target.exists():
            if hashlib.md5(target.read_bytes()).hexdigest() != EXPECTED[target.name]:
                raise ValueError(f"Existing file has unexpected content: {target}")
            print(target.name, "already verified")
            continue
        temporary = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(file["links"]["self"], timeout=90) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024*1024):
                handle.write(chunk)
        if hashlib.md5(temporary.read_bytes()).hexdigest() != EXPECTED[target.name]:
            raise ValueError(f"Download integrity check failed: {target.name}")
        temporary.rename(target)
        print(target.name, "downloaded and verified")
    (out / "zenodo_record.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
