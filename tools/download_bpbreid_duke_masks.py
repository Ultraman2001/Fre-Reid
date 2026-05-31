import argparse
import os
import shutil
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, build_opener


DUKE_MASKS_FILE_ID = "1N_YzhkEvDqIj2FEUyq-pFoMmtxTYgYtb"
DUKE_MASKS_NAME = "DukeMTMC-reID.zip"


def _get_confirm_token(response):
    for key, value in response.headers.items():
        if key.lower() == "set-cookie":
            cookies = value.split(";")
            for cookie in cookies:
                cookie = cookie.strip()
                if cookie.startswith("download_warning"):
                    return cookie.split("=", 1)[1]
    return None


def _download_google_drive_file(file_id, output_path, chunk_size=1024 * 1024):
    opener = build_opener()
    base_url = "https://drive.google.com/uc?export=download&id={}".format(file_id)

    response = opener.open(Request(base_url))
    token = _get_confirm_token(response)

    if token is not None:
        response.close()
        confirm_url = "{}&confirm={}".format(base_url, token)
        response = opener.open(Request(confirm_url))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with response, output_path.open("wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total // chunk_size != (total - len(chunk)) // chunk_size:
                print("\rDownloaded {:.1f} MB".format(total / 1024 / 1024), end="", flush=True)
    print()


def _find_masks_dir(extract_root):
    candidates = []
    for path in extract_root.rglob("pifpaf_maskrcnn_filtering"):
        if path.is_dir():
            candidates.append(path)
    if candidates:
        return candidates[0]

    for path in extract_root.rglob("pifpaf"):
        if path.is_dir():
            candidates.append(path)
    if candidates:
        return candidates[0]
    return None


def _copy_tree(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print("Destination exists, merging into {}".format(dst))
    shutil.copytree(src, dst, dirs_exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="Download BPBreID PifPaf human parsing labels for DukeMTMC-reID."
    )
    parser.add_argument(
        "--root-dir",
        default="/workspace/data",
        help="Dataset root containing dukemtmcreid. Default: /workspace/data",
    )
    parser.add_argument(
        "--dataset-name",
        default="dukemtmcreid",
        help="Fre-Reid Duke dataset folder name. Default: dukemtmcreid",
    )
    parser.add_argument(
        "--masks-dir",
        default="pifpaf_maskrcnn_filtering",
        help="Mask subdirectory name used by config. Default: pifpaf_maskrcnn_filtering",
    )
    parser.add_argument(
        "--archive",
        default="",
        help="Optional path to an already downloaded DukeMTMC-reID.zip.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded zip after extraction.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    dataset_dir = root_dir / args.dataset_name
    archive_path = Path(args.archive).expanduser().resolve() if args.archive else dataset_dir / DUKE_MASKS_NAME
    extract_root = dataset_dir / "_bpbreid_masks_extract"
    final_dir = dataset_dir / "masks" / args.masks_dir

    if not archive_path.exists():
        print("Downloading official BPBreID Duke human parsing labels:")
        print("  Google Drive file id: {}".format(DUKE_MASKS_FILE_ID))
        print("  Output archive: {}".format(archive_path))
        _download_google_drive_file(DUKE_MASKS_FILE_ID, archive_path)
    else:
        print("Using existing archive: {}".format(archive_path))

    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print("Extracting {} ...".format(archive_path))
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_root)

    masks_src = _find_masks_dir(extract_root)
    if masks_src is None:
        raise RuntimeError("Could not find pifpaf masks directory inside {}".format(archive_path))

    print("Installing masks:")
    print("  from: {}".format(masks_src))
    print("  to:   {}".format(final_dir))
    _copy_tree(masks_src, final_dir)

    shutil.rmtree(extract_root)
    if not args.keep_archive:
        archive_path.unlink(missing_ok=True)

    print("Done. Fre-Reid config should use:")
    print("  MASKS_BASE_DIR: 'masks'")
    print("  MASKS_DIR: '{}'".format(args.masks_dir))
    print("  MASKS_SUFFIX: '.npy'")


if __name__ == "__main__":
    main()
