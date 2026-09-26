"""Download one authorized Kaggle competition file with verified byte ranges.

The stock Kaggle CLI uses one streaming connection, which is impractically slow
on the current link for the multi-gigabyte Phase 2 payload.  This helper obtains
the official signed URL through the authenticated Kaggle SDK, opens a bounded
number of byte-range requests, verifies every Content-Range, and atomically
assembles the final file.  It never prints the signed URL or authentication
headers.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


CONTENT_RANGE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


def signed_download(competition: str, file_name: str) -> tuple[str, dict[str, str], int]:
    """Issue a signed URL without the CLI's extra OAuth introspection call."""
    token_path = Path.home() / ".kaggle" / "access_token"
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"Empty Kaggle access token: {token_path}")
    endpoint = (
        "https://www.kaggle.com/api/v1/"
        "competitions.CompetitionApiService/DownloadDataFile"
    )
    payload = {"competitionName": competition, "fileName": file_name}
    signed_url: str | None = None
    last_diagnostic = "not attempted"
    for attempt in range(1, 11):
        try:
            with requests.Session() as session:
                # The short Kaggle API request works more reliably through the
                # configured proxy. The large storage body is fetched direct.
                response = session.post(
                    endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {token}"},
                    timeout=(15, 30),
                    allow_redirects=False,
                )
                with response:
                    last_diagnostic = f"HTTP {response.status_code}"
                    response.raise_for_status()
                    if response.is_redirect:
                        signed_url = response.headers.get("Location")
                        if not signed_url:
                            raise ValueError("Redirect omitted Location")
                    else:
                        signed_url = str(response.json()["url"])
            break
        except (KeyError, ValueError, OSError, requests.RequestException) as exc:
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                last_diagnostic = f"HTTP {exc.response.status_code}"
            else:
                last_diagnostic = type(exc).__name__
            if attempt == 10:
                raise RuntimeError(
                    "Failed to issue Kaggle signed URL after 10 attempts; "
                    f"last result: {last_diagnostic}"
                ) from None
            time.sleep(min(attempt, 10))
    assert signed_url is not None

    # A one-byte direct request proves range support and gives the authoritative
    # total without exposing the signed URL in logs.
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            signed_url,
            headers={"Range": "bytes=0-0"},
            stream=True,
            timeout=(30, 60),
        )
        with response:
            if response.status_code != 206:
                raise RuntimeError(
                    f"Official storage range probe returned HTTP {response.status_code}, expected 206"
                )
            match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
            if match is None or (int(match.group(1)), int(match.group(2))) != (0, 0):
                raise RuntimeError("Official storage range probe returned an invalid Content-Range")
            size = int(match.group(3))
    return signed_url, {}, size


def ranges_for(size: int, workers: int) -> list[tuple[int, int]]:
    width = (size + workers - 1) // workers
    return [(start, min(size - 1, start + width - 1)) for start in range(0, size, width)]


def download_range(
    url: str,
    headers: dict[str, str],
    start: int,
    end: int,
    destination: Path,
    retries: int,
) -> dict[str, object]:
    expected = end - start + 1
    if destination.is_file() and destination.stat().st_size == expected:
        return {"start": start, "end": end, "bytes": expected, "reused": True}

    temporary = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(1, retries + 1):
        temporary.unlink(missing_ok=True)
        request_headers = dict(headers)
        request_headers["Range"] = f"bytes={start}-{end}"
        try:
            # Kaggle API authentication uses the configured proxy, but the
            # proxy truncates long Google Storage transfers on this host.
            # Signed storage URLs are public bearer URLs, so use a direct
            # session only for the range body after the authenticated URL has
            # been issued.
            with requests.Session() as session:
                session.trust_env = False
                response = session.get(
                    url,
                    headers=request_headers,
                    stream=True,
                    timeout=(30, 300),
                )
                with response:
                    if response.status_code != 206:
                        raise RuntimeError(
                            f"range {start}-{end}: HTTP {response.status_code}, expected 206"
                        )
                    match = CONTENT_RANGE.fullmatch(response.headers.get("Content-Range", ""))
                    if match is None or (int(match.group(1)), int(match.group(2))) != (start, end):
                        raise RuntimeError(
                            f"range {start}-{end}: invalid Content-Range "
                            f"{response.headers.get('Content-Range')!r}"
                        )
                    with temporary.open("wb") as handle:
                        for block in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if block:
                                handle.write(block)
            actual = temporary.stat().st_size
            if actual != expected:
                raise RuntimeError(f"range {start}-{end}: {actual} bytes, expected {expected}")
            os.replace(temporary, destination)
            return {"start": start, "end": end, "bytes": expected, "reused": False}
        except (OSError, requests.RequestException, RuntimeError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(
                    f"range {start}-{end} failed after {retries} attempts: "
                    f"{type(exc).__name__}"
                ) from None
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def assemble(parts: list[Path], output: Path, expected_size: int) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".assembling")
    with temporary.open("wb") as target:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
    if temporary.stat().st_size != expected_size:
        raise RuntimeError(
            f"assembled size {temporary.stat().st_size} does not match official {expected_size}"
        )
    digest = sha256_file(temporary)
    os.replace(temporary, output)
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    url, headers, size = signed_download(args.competition, args.file)
    ranges = ranges_for(size, args.workers)
    part_dir = args.output.parent / f".{args.output.name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_paths = [part_dir / f"part-{index:03d}" for index in range(len(ranges))]

    print({"file": args.file, "official_bytes": size, "ranges": len(ranges)})
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = {
            executor.submit(
                download_range,
                url,
                headers,
                start,
                end,
                part_paths[index],
                args.retries,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        for future in as_completed(futures):
            result = future.result()
            print({"completed_range": futures[future], **result}, flush=True)

    digest = assemble(part_paths, args.output, size)
    print({"output": str(args.output), "bytes": size, "sha256": digest})


if __name__ == "__main__":
    main()
