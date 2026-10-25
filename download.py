from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlsplit
import urllib.request
from tqdm import tqdm

URL_FILE = "urls.txt"
OUTPUT_DIR = Path("dataset")
MAX_WORKERS = 7
CHUNK_SIZE = 64 * 1024  # 64 KB


def get_filename(url: str) -> str:
    path = urlsplit(url).path
    name = Path(path).name
    return name if name else "downloaded_file"


def download_file(url: str, dest_dir: Path, position: int) -> None:
    filename = get_filename(url)
    dest_path = dest_dir / filename
    temp_path = dest_dir / f"{filename}.part"

    # Resume support: check existing bytes in partial download
    downloaded_bytes = temp_path.stat().st_size if temp_path.exists() else 0

    req = urllib.request.Request(url)
    if downloaded_bytes > 0:
        req.add_header("Range", f"bytes={downloaded_bytes}-")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = response.headers.get("Content-Length")
            total = (
                int(total_size) + downloaded_bytes
                if total_size
                else None
            )

            # 416 or already completed
            if response.status == 416:
                if temp_path.exists():
                    temp_path.replace(dest_path)
                return

            mode = "ab" if downloaded_bytes > 0 else "wb"
            with open(temp_path, mode) as f, tqdm(
                total=total,
                initial=downloaded_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename[:20],
                leave=False,
                position=position,
            ) as bar:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))

        # Rename upon clean finish
        temp_path.replace(dest_path)

    except Exception as e:
        print(f"\n[Error] {filename}: {e}")
        raise


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(URL_FILE, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    if not urls:
        print("No URLs found in urls.txt")
        return

    print(f"Downloading {len(urls)} files to ./{OUTPUT_DIR} (7 workers)...\n")

    # Main overall progress bar
    with tqdm(total=len(urls), desc="Overall Progress", unit="file", position=0) as overall_bar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Assign worker slot positions (1 to 7) for nested tqdm rendering
            futures = {
                executor.submit(download_file, url, OUTPUT_DIR, (i % MAX_WORKERS) + 1): url
                for i, url in enumerate(urls)
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    pass
                overall_bar.update(1)

    print(f"\nDone! Files saved in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()