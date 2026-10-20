"""Check local research links, footnotes, source index, and snapshot hashes.

Does not execute upstream code, load model weights, or certify game results.
Run with Python's standard library from any working directory.
"""

from pathlib import Path
import csv
import hashlib
import json
import re
import sys
from urllib.parse import unquote, urlsplit


def main():
    root = Path(__file__).resolve().parents[1]
    errors = []
    documents = [root / "README.md", *root.glob("20*-*.md")]
    for document in documents:
        text = document.read_text()
        refs = set(re.findall(r"\[\^([a-z0-9-]+)\](?!:)", text))
        definitions = re.findall(r"^\[\^([a-z0-9-]+)\]:", text, flags=re.M)
        for missing in sorted(refs - set(definitions)):
            errors.append(f"{document.name}: undefined footnote {missing}")
        if len(definitions) != len(set(definitions)):
            errors.append(f"{document.name}: duplicate footnote definitions")
        for target in re.findall(r"\[[^\]\n]+\]\(([^)\n]+)\)", text):
            target = target.strip("<>")
            if urlsplit(target).scheme or target.startswith("#"):
                continue
            path = document.parent / unquote(target.split("#")[0])
            if not path.exists():
                errors.append(f"{document.name}: missing local link {target}")
        if len(text.splitlines()) > 1000:
            errors.append(f"{document.name}: exceeds 1000 lines")

    manifest = json.loads((root / "sources/source-manifest.json").read_text())
    total = 0
    for repository in manifest:
        if not re.fullmatch(r"[a-f0-9]{40}", repository["commit"]):
            errors.append(f"{repository['id']}: invalid commit")
        for entry in repository["files"]:
            file = root / "sources" / entry["path"]
            if not file.exists():
                errors.append(f"snapshot missing: {file}")
                continue
            actual = hashlib.sha256(file.read_bytes()).hexdigest()
            if actual != entry["sha256"]:
                errors.append(f"snapshot hash mismatch: {file}")
            total += 1

    with (root / "sources/资料索引.tsv").open() as stream:
        sources = list(csv.DictReader(stream, delimiter="\t"))
    ids = [row["id"] for row in sources]
    if len(ids) != len(set(ids)):
        errors.append("duplicate source index IDs")
    for row in sources:
        if row["priority"] not in {"P0", "P1", "P2"}:
            errors.append(f"{row['id']}: invalid priority")
        if not row["url"].startswith("https://"):
            errors.append(f"{row['id']}: invalid URL")
        if row["local_path"] and not (root / row["local_path"]).exists():
            errors.append(f"{row['id']}: missing local source")

    for name in ("hf-model-api.json", "hf-chain.json", "hf-checkpoint-inspection.json"):
        json.loads((root / "sources" / name).read_text())

    if errors:
        print("\n".join(errors))
        return 1
    print(
        f"OK: {len(documents)} documents; {len(sources)} indexed links; "
        f"{len(manifest)} repositories / {total} snapshot hashes; 3 HF JSON artifacts."
    )
    print("Scope: local integrity only; external links and game behavior not tested.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
