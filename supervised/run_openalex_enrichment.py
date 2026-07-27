r"""Fetch OpenAlex author/reference IDs for local papers.

This enrichment deliberately stores identifiers and graph edges, not current
citation counts or current author metrics. Downstream feature scripts build
time-safe aggregates from the local train split only.

Example:
  D:\conda\envs\cs7641-team7\python.exe supervised/run_openalex_enrichment.py --cohort c1

For full c1 enrichment, set OPENALEX_API_KEY if available. The script resumes
from existing output and can be smoke-tested with --limit.
"""

import argparse
import gzip
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
OPENALEX_BASE = "https://api.openalex.org/works"
SELECT_FIELDS = "id,doi,publication_year,authorships,referenced_works,primary_location"


def read_local_records(path):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            rows.append(
                {
                    "arxiv_id": record.get("arxiv_id"),
                    "openalex_id": record.get("openalex_id"),
                    "doi": record.get("doi"),
                    "split": record.get("split"),
                    "year": record.get("year"),
                }
            )
    return rows


def normalize_openalex_work_id(value):
    if not value:
        return None
    value = str(value).strip()
    if not value:
        return None
    return value.rsplit("/", 1)[-1]


def normalize_openalex_url(value):
    if not value:
        return None
    value = str(value).strip()
    if value.startswith("https://openalex.org/"):
        return value
    if value.startswith("W"):
        return f"https://openalex.org/{value}"
    return value


def read_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            work_id = normalize_openalex_work_id(row.get("openalex_id"))
            if work_id:
                completed.add(work_id)
    return completed


def chunks(values, size):
    for idx in range(0, len(values), size):
        yield values[idx : idx + size]


def request_batch(work_ids, api_key=None, mailto=None, timeout=60):
    params = {
        "filter": "openalex_id:" + "|".join(work_ids),
        "select": SELECT_FIELDS,
        "per-page": len(work_ids),
    }
    if api_key:
        params["api_key"] = api_key
    if mailto:
        params["mailto"] = mailto
    url = OPENALEX_BASE + "?" + urlencode(params, safe=":|,")
    request = Request(url, headers={"User-Agent": "cs7641-team7-openalex-enrichment/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_work(work):
    author_ids = []
    institution_ids = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        author_id = author.get("id")
        if author_id:
            author_ids.append(author_id)
        for institution in authorship.get("institutions") or []:
            institution_id = institution.get("id")
            if institution_id:
                institution_ids.append(institution_id)

    referenced_works = [
        normalize_openalex_url(ref)
        for ref in (work.get("referenced_works") or [])
        if normalize_openalex_url(ref)
    ]
    source = ((work.get("primary_location") or {}).get("source") or {}).get("id")
    return {
        "openalex_id": normalize_openalex_url(work.get("id")),
        "publication_year": work.get("publication_year"),
        "doi": work.get("doi"),
        "author_ids": list(dict.fromkeys(author_ids)),
        "institution_ids": list(dict.fromkeys(institution_ids)),
        "referenced_work_ids": list(dict.fromkeys(referenced_works)),
        "source_id": source,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="c1")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--api-key", default=os.environ.get("OPENALEX_API_KEY"))
    parser.add_argument("--mailto", default=os.environ.get("OPENALEX_MAILTO"))
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data) if args.data else ROOT / "data" / f"{args.cohort}.jsonl.gz"
    out_path = (
        Path(args.out)
        if args.out
        else ART / f"openalex_{args.cohort}_author_ref_enrichment.jsonl.gz"
    )

    records = read_local_records(data_path)
    work_ids = [
        normalize_openalex_work_id(row["openalex_id"])
        for row in records
        if normalize_openalex_work_id(row.get("openalex_id"))
    ]
    work_ids = list(dict.fromkeys(work_ids))
    if args.limit is not None:
        work_ids = work_ids[: args.limit]

    completed = read_completed(out_path)
    pending = [work_id for work_id in work_ids if work_id not in completed]
    print(
        f"OpenAlex enrichment: total={len(work_ids):,}, "
        f"completed={len(completed):,}, pending={len(pending):,} -> {out_path}",
        flush=True,
    )
    if pending and not args.api_key:
        print(
            "No OPENALEX_API_KEY provided. A short smoke test may work, but full "
            "c1 enrichment usually needs an API key because list requests have a "
            "small unauthenticated daily budget.",
            flush=True,
        )

    t0 = time.time()
    fetched = 0
    with gzip.open(out_path, "at", encoding="utf-8") as out:
        for batch_num, batch in enumerate(chunks(pending, args.batch_size), start=1):
            for attempt in range(args.max_retries):
                try:
                    payload = request_batch(batch, api_key=args.api_key, mailto=args.mailto)
                    break
                except (HTTPError, URLError, TimeoutError) as exc:
                    wait = min(60, (2**attempt) * 2)
                    print(f"  request failed ({exc}); retrying in {wait}s", flush=True)
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Failed batch after {args.max_retries} retries: {batch[:3]}")

            by_id = {
                normalize_openalex_work_id(work.get("id")): extract_work(work)
                for work in payload.get("results", [])
            }
            for work_id in batch:
                row = by_id.get(work_id) or {
                    "openalex_id": f"https://openalex.org/{work_id}",
                    "missing_from_openalex": True,
                    "author_ids": [],
                    "institution_ids": [],
                    "referenced_work_ids": [],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                fetched += 1

            if batch_num % 10 == 0 or fetched == len(pending):
                print(
                    f"  fetched {fetched:,}/{len(pending):,} pending "
                    f"in {time.time() - t0:.0f}s",
                    flush=True,
                )
            time.sleep(args.sleep)

    print(f"Done -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
