import os
import shutil

BASE_DIR = "data"

TRAIN_FOLDERS = ["Fisheye8K", "LOAF", "Visdrone"]
TEST_FOLDERS = ["Fisheye1keval"]

def move_loaf(dest_dir):
    loaf_dir = os.path.join(BASE_DIR, "LOAF")
    if not os.path.exists(loaf_dir):
        print("LOAF folder not found.")
        return

    os.makedirs(os.path.join(dest_dir, "LOAF"), exist_ok=True)


    for item in ["train", "instances_train_converted.json"]:
        src = os.path.join(loaf_dir, item)
        dst = os.path.join(dest_dir, "LOAF", item)
        if os.path.exists(src):
            print(f"Moving LOAF/{item} to Train...")
            shutil.move(src, dst)
        else:
            print(f"LOAF item not found: {item}")

def move_fisheye8k(dest_dir):
    outer = os.path.join(BASE_DIR, "fisheye8k")
    inner = os.path.join(outer, "Fisheye8K")
    if os.path.exists(inner):
        print("Moving inner Fisheye8K to Train...")
        dst = os.path.join(dest_dir, "Fisheye8K")
        shutil.move(inner, dst)
    else:
        print("Inner Fisheye8K not found.")

def move_to_subfolder(subfolder_name, folder_names):
    subfolder_path = os.path.join(BASE_DIR, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    for folder in folder_names:
        if folder.lower() == "loaf":
            move_loaf(subfolder_path)
        elif folder.lower() == "fisheye8k":
            move_fisheye8k(subfolder_path)
        else:
            src = os.path.join(BASE_DIR, folder)
            dst = os.path.join(subfolder_path, folder)
            if os.path.exists(src):
                print(f"Moving {folder} to {subfolder_name}...")
                shutil.move(src, dst)
                print(f"{folder} moved to {subfolder_name}.")
            else:
                print(f"Folder not found: {folder}")

if __name__ == "__main__":
    print("\n Organizing dataset folders...")
    move_to_subfolder("Train", TRAIN_FOLDERS)
    move_to_subfolder("Test", TEST_FOLDERS)
    print("\n Dataset organization complete!")
