import os
import kagglehub
import gdown
import shutil
import zipfile
import tarfile
import time

BASE_DIR = "data"

DATASETS = {
    "fisheye8k": {
        "kaggle_slug": "flap1812/fisheye8k",
        "zip_name": None,
    },
    "fisheye1keval": {
        "kaggle_slug": "duongtran1909/fisheye1keval",
        "zip_name": None,
    },
    "loaf_images": {
        "gdrive_id": "1hb4RhaWrz4n6DRbuhw6yfgb8N0mWGPq6",
        "zip_name": "loaf_images.zip"
    },
    "loaf_annotations": {
        "gdrive_id": "1oA7aGpsmDSH99VHspR7gUr5a5sn9zexc",
        "zip_name": "loaf_annotations.zip"
    },
    "visdrone_raw": {
        "gdrive_id": "1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn",
        "zip_name": "visdrone_train.zip"
    }
}


def download_from_gdrive_with_retry(file_id, output, max_retries=3):
    for attempt in range(max_retries):
        try:
            gdown.download(id=file_id, output=output, quiet=False, fuzzy=True)
            return True
        except Exception as e:
            print(f"Retry {attempt + 1} failed: {e}")
            time.sleep(5)
    return False


def smart_unpack(file_path, dest):
    if zipfile.is_zipfile(file_path):
        shutil.unpack_archive(file_path, dest)
    elif tarfile.is_tarfile(file_path):
        shutil.unpack_archive(file_path, dest, format='tar')
    else:
        raise RuntimeError(f"Unsupported archive format or corrupted file: {file_path}")


def download_and_extract(name, info):
    dest = os.path.join(BASE_DIR, name)
    os.makedirs(dest, exist_ok=True)

    try:
        if "kaggle_slug" in info:
            print(f"\n Downloading {name} from Kaggle...")
            dataset_path = kagglehub.dataset_download(info["kaggle_slug"])
            shutil.copytree(dataset_path, dest, dirs_exist_ok=True)
        elif "gdrive_id" in info:
            zip_path = os.path.join(BASE_DIR, info["zip_name"])
            if not os.path.exists(zip_path):
                print(f"\n Downloading {name} from GDrive...")
                success = download_from_gdrive_with_retry(info["gdrive_id"], zip_path)
                if not success:
                    raise RuntimeError("GDrive download failed after retries.")
            print(f"Extracting {name}...")
            smart_unpack(zip_path, dest)
        print(f"Finished [{name}]")
    except Exception as e:
        print(f" Failed [{name}]: {e}")


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    for name, info in DATASETS.items():
        download_and_extract(name, info)
    print("\n DONE: All datasets attempted.")
