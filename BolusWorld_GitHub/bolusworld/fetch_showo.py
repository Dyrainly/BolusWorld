"""Fetch the official Show-o2 1.5B binary from ModelScope and verify its digest."""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen


URL = "https://modelscope.cn/api/v1/models/showlab/show-o2-1.5B/repo?Revision=master&FilePath=pytorch_model.bin"
SIZE = 5661862314
SHA256 = "a596cbc305c1df987c125d4f218e78f39b681621904cccfb2a3bf0ca0327f92c"
CHUNK = 32 * 2**20


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download(dest):
    import fcntl

    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".download.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        fetch_weights(dest)


def fetch_weights(dest):
    output = dest / "pytorch_model.bin"
    if output.exists():
        if digest(output) != SHA256:
            raise ValueError(f"Incorrect weight digest: {output}")
        return
    staging = dest / "modelscope_parts"
    staging.mkdir(exist_ok=True)

    def fetch(start):
        stop = min(start + CHUNK, SIZE)
        part = staging / f"{start:012}"
        if part.is_file() and part.stat().st_size == stop - start:
            return part
        tmp = part.with_suffix(".tmp")
        for attempt in range(8):
            try:
                request = Request(URL, headers={"Range": f"bytes={start}-{stop - 1}"})
                with urlopen(request, timeout=30) as response, tmp.open("wb") as stream:
                    if response.headers.get("Content-Range") != f"bytes {start}-{stop - 1}/{SIZE}":
                        raise ValueError("Invalid ModelScope response range")
                    for chunk in iter(lambda: response.read(2**20), b""):
                        stream.write(chunk)
                if tmp.stat().st_size != stop - start:
                    raise IOError("Incomplete range")
                tmp.replace(part)
                print(f"downloaded {start}-{stop - 1}", flush=True)
                return part
            except (OSError, ValueError) as exc:
                if attempt == 7:
                    raise
                print(f"retry {start} ({attempt + 1}): {exc}", flush=True)
                time.sleep(2 + attempt)

    tmp_output = output.with_suffix(".tmp")
    workers = int(os.environ.get("SHOWO_DOWNLOAD_WORKERS", "24"))
    with ThreadPoolExecutor(max_workers=workers) as pool, tmp_output.open("wb") as stream:
        for part in pool.map(fetch, range(0, SIZE, CHUNK)):
            with part.open("rb") as src:
                for chunk in iter(lambda: src.read(2**20), b""):
                    stream.write(chunk)
    if output.exists() and digest(output) == SHA256:
        tmp_output.unlink()
    else:
        if tmp_output.stat().st_size != SIZE or digest(tmp_output) != SHA256:
            raise ValueError("Show-o2 checkpoint failed integrity verification")
        os.replace(tmp_output, output)
    for part in staging.iterdir():
        if part.name.isdecimal():
            part.unlink()
    staging.rmdir()
    print(f"verified {output}", flush=True)


if __name__ == "__main__":
    import sys

    download(Path(sys.argv[1]))
