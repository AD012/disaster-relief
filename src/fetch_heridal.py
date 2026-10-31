"""Download one split of HERIDAL from an Ultralytics Platform NDJSON export.

Usage:
    python src/fetch_heridal.py --ndjson data/raw/heridal/heridal.ndjson --split test

Images go to data/raw/heridal/images/<split>/. The export's URLs are signed and
expire (about a week after export), so re-export from the platform if they fail.
Already-downloaded files are skipped.
"""
import argparse, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson", type=Path, default=Path("data/raw/heridal/heridal.ndjson"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.ndjson.read_text().splitlines() if l.strip()]
    todo = [r for r in rows if r["type"] == "image" and r["split"] == args.split]
    out_dir = args.ndjson.parent / "images" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    def fetch(r):
        path = out_dir / Path(r["file"]).name
        if path.exists() and path.stat().st_size > 0:
            return "skipped"
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(r["url"], tmp)
        tmp.rename(path)
        return "downloaded"

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(fetch, todo))
    print(f"{args.split}: {results.count('downloaded')} downloaded, {results.count('skipped')} skipped -> {out_dir}")


if __name__ == "__main__":
    main()
