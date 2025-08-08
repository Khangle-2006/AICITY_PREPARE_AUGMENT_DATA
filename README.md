## Data preparation

The dataset we use include: 
- `Fisheye8k`
- `LOAF`
- `Visdrone`
- `Fisheye1keval`

### Installation

**1. Download all required datasets by running:**
```
python .\Dataset\Download_dataset.py
```

**2. Preprocess datasets using the following command:**
```
python .\Dataset\restructure_and_process.py
```
This step includes several sub-tasks:

- **VisDrone:**
  - Converts annotations to **YOLO format** and maps each class to match those in the Fisheye8K dataset (other classes are ignored).
  - Saves YOLO annotations in the `labels/` folder.
  - Also generates a COCO-format `train.json` file under the `Visdrone/` folder.

- **LOAF:**
  - LOAF originally uses **rotated bounding boxes**.
  - This step converts them to **axis-aligned bounding boxes** and adds a `radius_point` field.
  - Result is saved as `instances_train_converted.json` under the `LOAF/` folder.

- **Fisheye1keval:**
  - Since this dataset is split into two separate folders, the script **merges** all images into a new folder named `Merged_images/` under `Fisheye1keval/`.

**3. Organize all datasets into `Train/` and `Test/` folders:**

```
python .\Dataset\Split_dataset.py    
```
---

### Dataset Structure
After processing, the `./data/` folder should look like this:
```bash
data/
├── Train/
│   ├── Fisheye8K/
│   │   ├── train/
│   │   └── test/
│   ├── LOAF/
│   │   ├── train/
│   │   └── instances_train_converted.json
│   └── Visdrone/
│       ├── annotations/
│       ├── images/
│       ├── labels/
│       └── train.json
└── Test/
    └── Fisheye1keval/
        └── Merged_images/

```
If you prefer to download the datasets manually, use the links below:
| **Dataset**       | **Links**                                                                 |
|-------------------|---------------------------------------------------------------------------|
| Fisheye8K         | [Fisheye8K](https://www.kaggle.com/datasets/flap1812/fisheye8k) |
| LOAF              | [LOAF ](https://loafisheye.github.io/download.html) |
| Visdrone          | [Visdrone ](https://github.com/VisDrone/VisDrone-Dataset) |
| Fisheye1keval     | [Fisheye1keval ](https://www.kaggle.com/datasets/duongtran1909/fisheye1keval) |



## Data Augmentation & Night Image Generation (Co-DETR + CycleGAN-Turbo)


We use **Co-DETR** to generate pseudo-labels for **VisDrone** and **Fisheye8K**.  
Additionally, **CycleGAN-Turbo** is used to generate synthetic **night-time images**.

---
### Run the following command:

```bash
python main.py \
    --codetr checkpoints/best_vis_fish_all.pth \
    --output_dir outputs \
    --device "cuda"
```

### Output Structure

After running the above command, the output will be organized like this:
```bash
outputs/
├── day/
│   ├── train/
│   │   ├── images/
│   │   └── train.json
│   └── test/
│       ├── images/
│       └── test.json
└── night/
    ├── train/
    │   ├── images/
    │   └── train.json
    └── test/
        ├── images/
        └── test.json
```

### Checkpoints
To reproduce results quickly, download pretrained checkpoints from the link below and place them in the ./checkpoints/ folder:
- [checkpoints](https://drive.google.com/drive/folders/1YxW_lkI8XrGQwToi9tFb6oMvcMPilBdJ?usp=sharing)