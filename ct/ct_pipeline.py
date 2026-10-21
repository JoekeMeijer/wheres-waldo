# Importing necessary libraries
import kfp
import argparse
import typing
from typing import Dict
from typing import NamedTuple
from kfp import dsl
from kfp.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component, 
                        OutputPath, 
                        InputPath)
import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.types import artifact_types
from kfp import compiler



#### Pipeline Component: Data Ingestion ####
@dsl.component(
        # only these three are third party packages
    packages_to_install=["pandas","google-cloud-storage", "lxml"],
    base_image="python:3.12.7-slim"
)
def download_data(project_id: str, 
                  bucket: str, 
                  file_name: str, 
                  dataset: Output[Dataset], 
                  images: Output[Dataset]):
    '''Download data and outputs the images and manifest'''
    # Importing necessary libraries
    from google.cloud import storage
    import pandas as pd
    import logging 
    import sys

    from pathlib import Path
    import re

    import tempfile
    import zipfile

    from dataclasses import dataclass
    from lxml import etree
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Regex experession variable that is used in helper functions
    SCENE_ID_PATTERN = re.compile(r"^(.*)_jpg\.rf\.[0-9a-f]+$")

    # Setting up folders within the VM to receive and unzip files
    work_dir = Path(tempfile.mkdtemp())
    zip_path = work_dir / "dataset.zip"
    extract_dir = work_dir / "extracted"
    extract_dir.mkdir()

    # Downloading the file from a Google bucket 
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket)
    blob = bucket.blob(file_name)
    blob.download_to_filename(str(zip_path))
    logging.info('Downloaded Data!')

    # Unzipping the file
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    logging.info("File Unzipped")

    # Helper functions and classes
    @dataclass
    class BoxAnnotation:
        label: str
        x1: int
        y1: int
        x2: int
        y2: int

        def as_tuple(self) -> tuple[int, int, int, int]:
            return (self.x1, self.y1, self.x2, self.y2)


    @dataclass
    class ImageAnnotation:
        filename: str
        width: int
        height: int
        boxes: list[BoxAnnotation]

    def parse_voc_xml(xml_path: str | Path) -> ImageAnnotation:
        """Parse one Pascal VOC XML file into an ImageAnnotation."""
        tree = etree.parse(str(xml_path))
        root = tree.getroot()

        filename = root.findtext("filename")
        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))

        boxes = []
        for obj in root.findall("object"):
            label = obj.findtext("name")
            bnd = obj.find("bndbox")
            boxes.append(
                BoxAnnotation(
                    label=label,
                    x1=int(round(float(bnd.findtext("xmin")))),
                    y1=int(round(float(bnd.findtext("ymin")))),
                    x2=int(round(float(bnd.findtext("xmax")))),
                    y2=int(round(float(bnd.findtext("ymax")))),
                )
            )

        return ImageAnnotation(filename=filename, width=width, height=height, boxes=boxes)

    def scene_id_from_stem(file_stem: str) -> str:
        """Recover the original scene id from a Roboflow-exported file stem.

        e.g. "10_15_4_jpg.rf.2df611cadb1226890b230a5b4d6d08e3" -> "10_15_4"
        Falls back to the full stem if the expected Roboflow suffix pattern
        isn't present (e.g. for images from a different source later).
        """
        match = SCENE_ID_PATTERN.match(file_stem)
        return match.group(1) if match else file_stem    

    def build_manifest(dataset_dir: str | Path) -> pd.DataFrame:
        """Scan train/valid/test folders and return one row per image:
        split, scene_id, image_path, xml_path, image width/height, and the
        list of Waldo boxes (in absolute pixel coordinates) found in it.
        """
        dataset_dir = Path(dataset_dir)
        rows = []

        for split in ("train", "valid", "test"):
            split_dir = dataset_dir / split
            if not split_dir.exists():
                continue

            for xml_path in sorted(split_dir.glob("*.xml")):
                ann = parse_voc_xml(xml_path)
                image_path = split_dir / ann.filename
                if not image_path.exists():
                    # Roboflow sometimes stores the extension case differently
                    candidates = list(split_dir.glob(xml_path.stem + ".*"))
                    image_path = next(p for p in candidates if p.suffix != ".xml")

                rows.append(
                    {
                        "split": split,
                        "scene_id": scene_id_from_stem(xml_path.stem),
                        "file_stem": xml_path.stem,
                        "image_path": str(image_path),
                        "xml_path": str(xml_path),
                        "width": ann.width,
                        "height": ann.height,
                        "num_boxes": len(ann.boxes),
                        "boxes": [b.as_tuple() for b in ann.boxes],
                    }
                )

        return pd.DataFrame(rows)

    # Building manifest from raw data, which is essentially the dataset containing the paths to the image files
    manifest = build_manifest(extract_dir / "wheres-waldo-voc")
    manifest.groupby("split").size()


    # Taking the first copy per scene since we only want ONE canonical image per scene
    clean_scene_ids = [str(i) for i in range(1, 20)]

    ### Note from author of training notebooks ###
    # Each clean scene has 1-3 copies from Roboflow's own train/valid/test split (the
    # valid/test ones are single, unaugmented copies; several train ones have 3
    # Roboflow-augmented copies). We only want ONE canonical image per scene: which
    # specific copy we pick doesn't matter much since we'll run our own augmentation
    # pipeline in notebook 02 regardless, so we just take the first copy per scene.
    ######

    # Code to get a cleaned manifest dataset
    clean = manifest[manifest["scene_id"].isin(clean_scene_ids)].copy()
    clean_manifest = (
        clean.sort_values("file_stem").groupby("scene_id", as_index=False).first()
    )
    clean_manifest = clean_manifest.sort_values(by="scene_id", key=lambda s: s.astype(int)).reset_index(drop=True)

    logging.info(f"Clean manifest: {len(clean_manifest)} scenes (was {manifest['scene_id'].nunique()} raw scenes)")

    # Storing the relevant images in a zip file that can be passed on to the next component
    with zipfile.ZipFile(images.path, "w") as zf:
        for p in clean_manifest["image_path"]:
            # We do not want the full path from the temp file, since the current temp folder will be discarded once the component is done running
            zf.write(p, arcname=Path(p).name)

    # Storing only the image path names in clean_manifest so that the images can be accessed in the next components
    clean_manifest["image_name"] = clean_manifest["image_path"].apply(lambda p: Path(p).name)

    # We no longer work with the XML files, and images now only have the path names instead of the full paths
    manifest_to_save = clean_manifest.drop(columns=["image_path", "xml_path"]).copy()
    # Saving manifest to a csv to be exported out of the component
    manifest_to_save["boxes"] = manifest_to_save["boxes"].apply(str)
    manifest_to_save.to_csv(dataset.path, index=False)



#### Pipeline Component: Preprocessing Data ####

@dsl.component(
    packages_to_install=['pandas', 'scikit-learn==1.3.2', "numpy",
                         "pillow", "google-cloud-storage", "albumentations==2.0.8"],
    base_image="python:3.12.7-slim"
)
def preprocess_data(project_id: str,
                    bucket: str, 
                    # For the hey_waldo augmentation, zip file from bucket
                    hey_waldo_blob: str,
                    # Manifest from previous component
                    in_manifest: Input[Dataset], 
                    # Zip file of images stored from previous component
                    in_images: Input[Dataset],
                    # Expanded patch manifest to be used in model training
                    out_exp_manifest: Output[Dataset],
                    # Fold assignment needed in training
                    out_fold_assignment: Output[Dataset], 
                    # Patch zip file needed for training
                    out_patches: Output[Dataset]):
    '''Preprocess data, outputs the patches, expanded patch manifest and fold assignment'''
    # From Lab
    from google.cloud import storage
    import logging 
    import sys

    # For the component
    import ast
    import sys
    from pathlib import Path
    import re

    # For unzipping
    import tempfile
    import zipfile

    PROJECT_ROOT = Path.cwd().parent

    sys.path.insert(0, str(PROJECT_ROOT))

    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import KFold


    from dataclasses import dataclass

    import albumentations as A

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  

    # Setting up folders in the VM
    work_dir = Path(tempfile.mkdtemp())
    IMAGES_DIR = work_dir / "images"
    PATCHES_DIR = work_dir / "patches"
    HEY_WALDO_DIR = work_dir / "hey_waldo_zip"
    OUT_DIR = PATCHES_DIR / "hey_waldo"

    # Turning these paths into directories within the parent temp folder
    for directory in (IMAGES_DIR, PATCHES_DIR, HEY_WALDO_DIR):
        directory.mkdir()

    # Unzipping files
    ## First, unzip the relevant images from the previous component
    with zipfile.ZipFile(in_images.path) as zf:
        zf.extractall(IMAGES_DIR)

    ## Next, download the hey-waldo images from the GCP bucket and unzip it
    client = storage.Client(project=project_id)
    hey_waldo_zip = work_dir / "hey_waldo.zip"
    client.bucket(bucket).blob(hey_waldo_blob).download_to_filename(str(hey_waldo_zip))
    with zipfile.ZipFile(hey_waldo_zip) as zf:
        zf.extractall(HEY_WALDO_DIR)

    # Ensuring that the unzipped files contains the files we need
    HEY_WALDO_ROOT = HEY_WALDO_DIR
    for cand in [HEY_WALDO_DIR, *HEY_WALDO_DIR.rglob("*")]:
        # If the candidate is a directory and the directory contains another directory in the given tuple
        if cand.is_dir() and any((cand / s).is_dir() for s in ("64", "128", "256")):
            # Then the candidate is the file we need for the functions below to run
            HEY_WALDO_ROOT = cand
            break
    

    # Helper functions and classes


    ## IoU
    def box_area(box: np.ndarray) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


    def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Intersection-over-union of two (x1, y1, x2, y2) boxes.

        IoU = area(A ∩ B) / area(A ∪ B). It's 0 for boxes that don't overlap and 1
        for identical boxes, which is what makes it a good scale-invariant measure
        of "did the model point at the right place" for both NMS (suppressing
        duplicate detections of the same object) and evaluation (is a predicted
        box close enough to the ground truth to count as a correct detection).
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = box_area(box_a) + box_area(box_b) - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    ## Patch generation
    PATCH_SIZE = 64  # fixed input size for the stage-1 CNN classifier
    NEGATIVE_IOU_THRESHOLD = 0.1  # a crop overlapping a box by more than this isn't a clean negative
    NEGATIVES_PER_SCENE = 40
    POSITIVE_CROPS_PER_BOX = 6  # jittered positive crops per ground-truth box


    @dataclass
    class Patch:
        image: np.ndarray  # HxWx3 uint8, already resized to PATCH_SIZE
        label: int  # 1 = waldo, 0 = not waldo
        scene_id: str
        source_box: tuple[int, int, int, int] | None  # None for negatives


    def _crop_and_resize(image: Image.Image, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        crop = image.crop((x1, y1, x2, y2)).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
        return np.array(crop.convert("RGB"))


    def generate_positive_patches(
        image: Image.Image, boxes: list[tuple[int, int, int, int]], scene_id: str, rng: np.random.Generator
    ) -> list[Patch]:
        """For each ground-truth box, take several jittered square crops around it.

        Jittering (random padding + a small center offset) is the positive-class
        augmentation: it teaches the classifier that Waldo doesn't have to be
        perfectly centered or tightly framed in a window to count as a match,
        which matters because a real sliding window will rarely land exactly on
        the box.
        """
        patches = []
        for box in boxes:
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            side = max(bw, bh)

            for _ in range(POSITIVE_CROPS_PER_BOX):
                pad = rng.uniform(0.15, 0.6) * side
                jitter_x = rng.uniform(-0.2, 0.2) * side
                jitter_y = rng.uniform(-0.2, 0.2) * side
                cx, cy = (x1 + x2) / 2 + jitter_x, (y1 + y2) / 2 + jitter_y
                half = side / 2 + pad

                crop_box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
                patches.append(
                    Patch(
                        image=_crop_and_resize(image, crop_box),
                        label=1,
                        scene_id=scene_id,
                        source_box=box,
                    )
                )
        return patches


    def generate_negative_patches(
        image: Image.Image,
        boxes: list[tuple[int, int, int, int]],
        scene_id: str,
        rng: np.random.Generator,
        n: int = NEGATIVES_PER_SCENE,
    ) -> list[Patch]:
        """Sample random square crops that don't meaningfully overlap any Waldo box.

        Crop sizes are drawn from a range around typical box sizes rather than a
        single fixed size, since a real sliding window will be run at multiple
        scales (notebook 03) and the classifier needs negative examples at those
        scales too, not just one.
        """
        W, H = image.width, image.height
        boxes_arr = [np.array(b) for b in boxes]

        patches = []
        attempts = 0
        while len(patches) < n and attempts < n * 20:
            attempts += 1
            side = rng.integers(20, 120)
            if side >= min(W, H):
                continue
            x1 = rng.integers(0, W - side)
            y1 = rng.integers(0, H - side)
            candidate = np.array([x1, y1, x1 + side, y1 + side])

            if boxes_arr and max(iou(candidate, b) for b in boxes_arr) > NEGATIVE_IOU_THRESHOLD:
                continue

            patches.append(
                Patch(
                    image=_crop_and_resize(image, tuple(candidate)),
                    label=0,
                    scene_id=scene_id,
                    source_box=None,
                )
            )
        return patches

    ## Augmentation
    def build_train_augmentation() -> A.Compose:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=25, border_mode=0, p=0.5),
                A.RandomResizedCrop(
                    size=(PATCH_SIZE, PATCH_SIZE), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.7
                ),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
                A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            ]
        )


    def augment_patch(image: np.ndarray, augmenter: A.Compose) -> np.ndarray:
        return augmenter(image=image)["image"]

    # Loading in the manifest generated from the previous component
    manifest = pd.read_csv(in_manifest.path)
    manifest["boxes"] = manifest["boxes"].apply(ast.literal_eval)
    manifest["scene_id"] = manifest["scene_id"].astype(str)

    ## Scene level k-fold split
    N_FOLDS = 5
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    scene_ids = manifest["scene_id"].to_numpy()
    fold_rows = []
    for fold_idx, (_, val_idx) in enumerate(kfold.split(scene_ids)):
        for scene_id in scene_ids[val_idx]:
            fold_rows.append({"scene_id": scene_id, "val_fold": fold_idx})

    fold_assignment = pd.DataFrame(fold_rows).sort_values("scene_id", key=lambda s: s.astype(int))

    # Exporting the fold assignment to be used in the model component
    fold_assignment.to_csv(out_fold_assignment.path, index=False)

    logging.info(f"Number of scenes in each fold: {fold_assignment["val_fold"].value_counts().sort_index()}")

    ## Generating the base patch corpus
    def materialize_patches(manifest: pd.DataFrame, out_dir: Path, seed: int = 0) -> pd.DataFrame:
        (out_dir / "positive").mkdir(parents=True, exist_ok=True)
        (out_dir / "negative").mkdir(parents=True, exist_ok=True)

        records = []
        for _, row in manifest.iterrows():
            rng = np.random.default_rng(seed + int(row["scene_id"]))
            # Since it is no longer the paths but the path names that are stored, we need to prefix the temp folder path
            image = Image.open(IMAGES_DIR / row["image_name"])

            all_patches = generate_positive_patches(image, row["boxes"], row["scene_id"], rng)
            all_patches += generate_negative_patches(image, row["boxes"], row["scene_id"], rng)

            for i, patch in enumerate(all_patches):
                label_name = "positive" if patch.label == 1 else "negative"
                filename = f"{row['scene_id']}_{label_name}_{i:03d}.png"
                path = out_dir / label_name / filename
                Image.fromarray(patch.image).save(path)
                # Storing the path to the patches relative to the root of the patches folder for easier reconstruction in the model step
                records.append({"scene_id": row["scene_id"], "label": patch.label, "path": path.relative_to(out_dir)})

        return pd.DataFrame(records)

    
    # Creating the baseline patches manifest dataset
    patch_manifest = materialize_patches(manifest, PATCHES_DIR)

    logging.info(f"Number of Patches containing Wally: {patch_manifest["label"].value_counts()}")

    # Fixing data-volume problem
    ## First we need create the necessary functions

    SCENE_ID_PATTERN = re.compile(r"^(\d+)_")

    def _scene_id_from_filename(filename: str) -> str:
        match = SCENE_ID_PATTERN.match(filename)
        if not match:
            raise ValueError(f"Unexpected Hey-Waldo filename format: {filename}")
        return match.group(1)


    def build_hey_waldo_manifest(
        scales: tuple[str, ...] = ("64", "128", "256"),
        max_negatives: int = 3000,
        seed: int = 0,
    ) -> pd.DataFrame:
        """Resize Hey-Waldo patches to PATCH_SIZE and materialize them under
        data/processed/patches/hey_waldo/, returning a manifest with the same
        columns as patch_manifest.csv (scene_id, label, path) so it can simply
        be concatenated with it.
        """
        rng = np.random.default_rng(seed)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

        positive_records = []
        negative_records = []

        for scale in scales:
            for label_name, label in (("waldo", 1), ("notwaldo", 0)):
                src_dir = HEY_WALDO_ROOT / scale / label_name
                if not src_dir.exists():
                    continue
                for src_path in src_dir.iterdir():
                    if not src_path.is_file():
                        continue
                    scene_id = _scene_id_from_filename(src_path.name)
                    record = {"scene_id": scene_id, "label": label, "_scale": scale, "_src": src_path}
                    (positive_records if label == 1 else negative_records).append(record)

        # Hey-Waldo's patch folders also include a couple of scenes (ids 20, 21)
        # that aren't among our 19 verified clean scenes and have no fold
        # assignment — drop them rather than risk an unaccounted-for leak.
        known_scene_ids = {str(i) for i in range(1, 20)}
        positive_records = [r for r in positive_records if r["scene_id"] in known_scene_ids]
        negative_records = [r for r in negative_records if r["scene_id"] in known_scene_ids]

        if len(negative_records) > max_negatives:
            keep_idx = rng.choice(len(negative_records), size=max_negatives, replace=False)
            negative_records = [negative_records[i] for i in keep_idx]

        rows = []
        for record in positive_records + negative_records:
            label_name = "waldo" if record["label"] == 1 else "notwaldo"
            out_name = f"{record['scene_id']}_{record['_scale']}_{label_name}_{record['_src'].name}"
            out_path = OUT_DIR / out_name

            img = Image.open(record["_src"]).convert("RGB").resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR)
            img.save(out_path)

            # We also have to make the paths relative here
            rows.append({"scene_id": record["scene_id"], "label": record["label"], "path": out_path.relative_to(PATCHES_DIR)})

        return pd.DataFrame(rows)


    # Expanding the patch corpus with Hey-Waldo
    hey_waldo_manifest = build_hey_waldo_manifest(max_negatives=3000, seed=0)
    hey_waldo_manifest["scene_id"] = hey_waldo_manifest["scene_id"].astype(str)
    logging.info(f"Hey-Waldo patches: {len(hey_waldo_manifest)} ({hey_waldo_manifest['label'].value_counts().to_dict()})")

    expanded_patch_manifest = pd.concat([patch_manifest, hey_waldo_manifest], ignore_index=True)
    # Writing the expanded manifest to a csv to be used in the model component
    expanded_patch_manifest.to_csv(out_exp_manifest.path, index=False)

    logging.info(f"\nOriginal patch corpus: {len(patch_manifest)} ({patch_manifest['label'].value_counts().to_dict()})")
    logging.info(f"Expanded patch corpus: {len(expanded_patch_manifest)} ({expanded_patch_manifest['label'].value_counts().to_dict()})")

    # We want to carry our patch images into the next component. We use a zipfile for this
    with zipfile.ZipFile(out_patches.path, "w", zipfile.ZIP_STORED) as zf:
        # For each image file in the folders within PATCHES
        for p in PATCHES_DIR.rglob("*"):
            # Only write if they are files
            if p.is_file():
                # We do not want to store the temp folder details, only the relative paths to a PATCHES_DIR later on
                zf.write(p, arcname=p.relative_to(PATCHES_DIR))



#### Pipeline Component: Training the CNN Model ####


@dsl.component(
    packages_to_install=['pandas', "pillow", "numpy",
                         "torch", "albumentations==2.0.8"],
    base_image="python:3.12.7-slim"
)
def train_cnn(in_patch_manifest: Input[Dataset], # Comes from expanded_patch_manifest
               # Comes from the fold assignments
               in_fold_assignment: Input[Dataset],
               # Comes from manifest_to_save
               in_scenes: Input[Dataset], 
               # Zip file of images
               in_images: Input[Dataset],
               # Zip file of patches
               in_patches: Input[Dataset],
               # Model Object
               out_model: Output[Model],
               # Metrics as a pandas dataset for comparability to current model metric in model repo
               out_metrics: Output[Dataset],
               # Dataset of individual scene results
               out_scene_results: Output[Dataset]):
    '''Train a CNN model with parameters taken from the notebook'''
    # From lab
    import logging

    # Importing the necessary packages
    import ast
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path.cwd().parent

    sys.path.insert(0, str(PROJECT_ROOT))

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image

    from dataclasses import dataclass, field

    import torch
    from torch import nn
    from torch.utils.data import Dataset as TorchDataset
    from torch.utils.data import DataLoader

    import albumentations as A

    # For unzipping
    import tempfile
    import zipfile

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)  


    PATCH_SIZE = 64  # fixed input size for the stage-1 CNN classifier
    # Seed for pytorch
    torch.manual_seed(0)

    IOU_THRESHOLD = 0.3  # used throughout this notebook to decide if a predicted box "counts" as a match

    # Unzipping zip fiiles

    # Make a temporary directory to store the unzipped files
    work_dir = Path(tempfile.mkdtemp())
    IMAGES_DIR = work_dir / "images"
    PATCHES_DIR = work_dir/ "patches"
    # Both of these need to be initalised as folders
    IMAGES_DIR.mkdir()
    PATCHES_DIR.mkdir()
    # Unzip zip files
    with zipfile.ZipFile(in_images.path) as zf:
        zf.extractall(IMAGES_DIR)
    with zipfile.ZipFile(in_patches.path) as zf:
        zf.extractall(PATCHES_DIR)

    # Defining functions to be used
    # IOU
    def box_area(box: np.ndarray) -> float:
        x1, y1, x2, y2 = box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


    def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Intersection-over-union of two (x1, y1, x2, y2) boxes.

        IoU = area(A ∩ B) / area(A ∪ B). It's 0 for boxes that don't overlap and 1
        for identical boxes, which is what makes it a good scale-invariant measure
        of "did the model point at the right place" for both NMS (suppressing
        duplicate detections of the same object) and evaluation (is a predicted
        box close enough to the ground truth to count as a correct detection).
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = box_area(box_a) + box_area(box_b) - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    # Metrics
    @dataclass
    class DetectionMatch:
        true_positives: int
        false_positives: int
        false_negatives: int
        localization_errors: list[float]  # center-distance (px) for matched TPs


    def match_detections(
        pred_boxes: list[tuple[int, int, int, int]],
        gt_boxes: list[tuple[int, int, int, int]],
        iou_threshold: float = 0.3,
    ) -> DetectionMatch:
        """Greedily match predicted boxes to ground-truth boxes.

        A predicted box counts as a true positive if its IoU with some
        not-yet-matched ground-truth box exceeds `iou_threshold` (each ground
        truth box can only be matched once, so duplicate detections of the same
        Waldo don't count as extra true positives).
        """
        matched_gt = set()
        tp = 0
        localization_errors = []

        for pred in pred_boxes:
            best_iou, best_gt_idx = 0.0, -1
            for gt_idx, gt in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                score = iou(np.array(pred), np.array(gt))
                if score > best_iou:
                    best_iou, best_gt_idx = score, gt_idx

            if best_iou >= iou_threshold:
                tp += 1
                matched_gt.add(best_gt_idx)
                gt = gt_boxes[best_gt_idx]
                pred_center = ((pred[0] + pred[2]) / 2, (pred[1] + pred[3]) / 2)
                gt_center = ((gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2)
                localization_errors.append(float(np.hypot(*(np.array(pred_center) - np.array(gt_center)))))

        fp = len(pred_boxes) - tp
        fn = len(gt_boxes) - len(matched_gt)
        return DetectionMatch(true_positives=tp, false_positives=fp, false_negatives=fn, localization_errors=localization_errors)


    def precision_recall(matches: list[DetectionMatch]) -> tuple[float, float]:
        tp = sum(m.true_positives for m in matches)
        fp = sum(m.false_positives for m in matches)
        fn = sum(m.false_negatives for m in matches)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return precision, recall


    def mean_localization_error(matches: list[DetectionMatch]) -> float | None:
        errors = [e for m in matches for e in m.localization_errors]
        return float(np.mean(errors)) if errors else None

    # Patch classifier
    """A small CNN for the stage-1 "is this 64x64 crop Waldo?" classifier.

    Kept deliberately small: with only 886 base patches (19 source scenes), a
    large network would overfit almost immediately. Four conv blocks with
    batchnorm + dropout, global average pooling instead of a big flatten+FC head
    (fewer parameters, less to overfit), one logit output for BCEWithLogitsLoss.
    """

    class PatchClassifier(nn.Module):
        def __init__(self, dropout: float = 0.3):
            super().__init__()
            self.features = nn.Sequential(
                self._conv_block(3, 16),
                self._conv_block(16, 32),
                self._conv_block(32, 64),
                self._conv_block(64, 128),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        @staticmethod
        def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self.pool(x).flatten(1)
            return self.classifier(x).squeeze(-1)  # raw logits, shape (B,)

    # To tensor
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


    def to_tensor(image: np.ndarray) -> torch.Tensor:
        normalized = (image.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(normalized.transpose(2, 0, 1)).float()

    # non_max_supressions
    def non_max_suppression(
        boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.3
    ) -> list[int]:
        """Greedy NMS: collapse a pile of overlapping candidate boxes into one
        box per real object.

        A sliding-window classifier fires on many overlapping windows around the
        same true Waldo (shifted by a few pixels, at several scales), each with
        its own confidence score. Non-max suppression is the standard fix:

        1. Take the highest-scoring remaining box.
        2. Keep it, and discard every other remaining box that overlaps it by
        more than `iou_threshold` (i.e. is almost certainly the same
        detection, not a different object).
        3. Repeat until no boxes remain.

        Returns the indices (into `boxes`/`scores`) of the boxes that survive,
        ordered from highest to lowest score.
        """
        boxes = np.asarray(boxes, dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)

        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            current = order[0]
            keep.append(int(current))

            if order.size == 1:
                break

            rest = order[1:]
            current_iou = np.array([iou(boxes[current], boxes[i]) for i in rest])
            order = rest[current_iou <= iou_threshold]

        return keep

    # Sliding Window
    DEFAULT_WINDOW_SIZES = (32, 48, 64, 96, 128)
    DEFAULT_STRIDE_FRACTION = 0.5
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


    @dataclass
    class Detection:
        box: tuple[int, int, int, int]
        score: float


    def windows_for_scale(width: int, height: int, size: int, stride: int) -> list[tuple[int, int, int, int]]:
        boxes = []
        for y in range(0, max(1, height - size + 1), stride):
            for x in range(0, max(1, width - size + 1), stride):
                boxes.append((x, y, x + size, y + size))
        return boxes


    @torch.no_grad()
    def detect(
        model: nn.Module,
        image: Image.Image,
        window_sizes: tuple[int, ...] = DEFAULT_WINDOW_SIZES,
        stride_fraction: float = DEFAULT_STRIDE_FRACTION,
        score_threshold: float = 0.5,
        nms_iou_threshold: float = 0.2,
        batch_size: int = 256,
        device: str = DEFAULT_DEVICE,
    ) -> list[Detection]:
        model.to(device)
        model.eval()
        W, H = image.width, image.height

        all_boxes: list[tuple[int, int, int, int]] = []
        for size in window_sizes:
            stride = max(1, int(size * stride_fraction))
            all_boxes.extend(windows_for_scale(W, H, size, stride))

        all_scores = np.zeros(len(all_boxes), dtype=np.float32)
        for start in range(0, len(all_boxes), batch_size):
            batch_boxes = all_boxes[start : start + batch_size]
            crops = [
                image.crop(b).resize((PATCH_SIZE, PATCH_SIZE), Image.BILINEAR) for b in batch_boxes
            ]
            tensors = torch.stack([to_tensor(np.array(c.convert("RGB"))) for c in crops]).to(device)
            logits = model(tensors)
            scores = torch.sigmoid(logits).cpu().numpy()
            all_scores[start : start + len(batch_boxes)] = scores

        keep_mask = all_scores >= score_threshold
        kept_boxes = np.array(all_boxes)[keep_mask]
        kept_scores = all_scores[keep_mask]

        if len(kept_boxes) == 0:
            return []

        keep_idx = non_max_suppression(kept_boxes, kept_scores, iou_threshold=nms_iou_threshold)
        return [Detection(box=tuple(kept_boxes[i]), score=float(kept_scores[i])) for i in keep_idx]


    # Augmentation.py file
    def build_train_augmentation() -> A.Compose:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=25, border_mode=0, p=0.5),
                A.RandomResizedCrop(
                    size=(PATCH_SIZE, PATCH_SIZE), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.7
                ),
                A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=0.7),
                A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
            ]
        )


    def augment_patch(image: np.ndarray, augmenter: A.Compose) -> np.ndarray:
        return augmenter(image=image)["image"]

    # Patch Dataset Class
    class PatchDataset(TorchDataset):
        def __init__(
            self,
            patch_manifest: pd.DataFrame,
            fold_assignment: pd.DataFrame,
            val_fold: int,
            split: str,
            # We now also need the path for where the patches are stored
            patches_root,
            augmenter: A.Compose | None = None,
        ):
            assert split in ("train", "val")
            merged = patch_manifest.merge(fold_assignment, on="scene_id", how="left")
            is_val_scene = merged["val_fold"] == val_fold
            self.rows = (merged[is_val_scene] if split == "val" else merged[~is_val_scene]).reset_index(
                drop=True
            )
            self.split = split
            self.patches_root = Path(patches_root)
            self.augmenter = augmenter if augmenter is not None else build_train_augmentation()

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            row = self.rows.iloc[idx]
            # Patches path names need to be prefixed with the root of the patches folder since they are relative
            image = np.array(Image.open(self.patches_root / row["path"]).convert("RGB"))
            if self.split == "train":
                image = augment_patch(image, self.augmenter)
            return to_tensor(image), torch.tensor(float(row["label"]))


    # Functions and classes needed for train_fold
    @dataclass
    class FoldHistory:
        train_loss: list[float] = field(default_factory=list)
        val_loss: list[float] = field(default_factory=list)
        val_accuracy: list[float] = field(default_factory=list)


    def run_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer=None, device: str = DEFAULT_DEVICE):
        is_train = optimizer is not None
        model.train(is_train)

        total_loss, correct, n = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            with torch.set_grad_enabled(is_train):
                logits = model(images)
                loss = criterion(logits, labels)
                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += ((torch.sigmoid(logits) > 0.5).float() == labels).sum().item()
            n += len(labels)

        return total_loss / n, correct / n

    # Train fold
    def train_fold(
        model: nn.Module,
        patch_manifest,
        fold_assignment,
        val_fold: int,
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = DEFAULT_DEVICE,
    ) -> FoldHistory:
        model.to(device)

        # The patches folder root directory has to be added
        train_ds = PatchDataset(patch_manifest, fold_assignment, val_fold, split="train", patches_root=PATCHES_DIR)
        val_ds = PatchDataset(patch_manifest, fold_assignment, val_fold, split="val", patches_root=PATCHES_DIR)

        # pin_memory + multiple workers only help when copying to a GPU; on CPU
        # they add overhead for no benefit.
        loader_kwargs = {"num_workers": 2, "pin_memory": True} if device != "cpu" else {}
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

        # Positive patches are far rarer than negative ones even after undersampling
        # (~126 positive vs ~760 negative base patches) — weight the loss so the
        # optimizer doesn't just learn to predict "negative" for everything.
        n_pos = (train_ds.rows["label"] == 1).sum()
        n_neg = (train_ds.rows["label"] == 0).sum()
        pos_weight = torch.tensor(n_neg / max(1, n_pos), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        history = FoldHistory()
        for _ in range(epochs):
            train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.val_accuracy.append(val_acc)

        return history

    logging.info(f"Training device: {DEFAULT_DEVICE}")

    ## Loading the patch data and the fold assignment data
    # Retrained on the expanded corpus (notebook 02, Part 2): original generated
    # patches + ~3,100 Hey-Waldo human-curated patches, since the honest result
    # further down used to be near-zero precision from too little patch diversity.
    # in_patch_manifest is the same as expanded_patch_manifest
    patch_manifest = pd.read_csv(in_patch_manifest.path)
    patch_manifest["scene_id"] = patch_manifest["scene_id"].astype(str)
    fold_assignment = pd.read_csv(in_fold_assignment.path)
    fold_assignment["scene_id"] = fold_assignment["scene_id"].astype(str)

    # in_scenes is the same as manifest_to_save
    scenes = pd.read_csv(in_scenes.path)
    scenes["boxes"] = scenes["boxes"].apply(ast.literal_eval)
    scenes["scene_id"] = scenes["scene_id"].astype(str)

    N_FOLDS = fold_assignment["val_fold"].nunique()
    logging.info(f"{len(patch_manifest)} patches, {len(scenes)} scenes, {N_FOLDS} folds")

    # Training the first fold
    model_fold0 = PatchClassifier()
    history_fold0 = train_fold(model_fold0, patch_manifest, fold_assignment, val_fold=0, epochs=25)

    # Training the model with full cross-validation
    fold_models = {0: model_fold0}
    all_matches = []
    per_scene_results = []
    # From notebook of the author
    SCORE_THRESHOLD = 0.6


    for fold in range(N_FOLDS):
        if fold == 0:
            fold_model = model_fold0
        else:
            fold_model = PatchClassifier()
            train_fold(fold_model, patch_manifest, fold_assignment, val_fold=fold, epochs=15)
            fold_models[fold] = fold_model

        val_scenes = scenes.merge(fold_assignment, on="scene_id").query("val_fold == @fold")
        for _, scene in val_scenes.iterrows():
            # Image paths need to be prefixed with the image root directory since they are only the image names now
            image = Image.open(IMAGES_DIR / scene["image_name"])
            dets = detect(fold_model, image, score_threshold=SCORE_THRESHOLD, nms_iou_threshold=0.2)
            pred_boxes = [d.box for d in dets]
            match = match_detections(pred_boxes, scene["boxes"], iou_threshold=IOU_THRESHOLD)
            all_matches.append(match)
            per_scene_results.append(
                {
                    "scene_id": scene["scene_id"],
                    "fold": fold,
                    "tp": match.true_positives,
                    "fp": match.false_positives,
                    "fn": match.false_negatives,
                }
            )
        logging.info(f"fold {fold} done")

    # Saving the metrics and other useful information
    precision, recall = precision_recall(all_matches)
    mle = mean_localization_error(all_matches)
    logging.info(f"\nAcross all {len(scenes)} scenes (IoU>={IOU_THRESHOLD}, score_threshold={SCORE_THRESHOLD}):")
    logging.info(f"  Precision: {precision:.3f}")
    logging.info(f"  Recall:    {recall:.3f}")
    logging.info(f"  Mean localization error (matched detections): {mle:.1f}px" if mle else "  No true positives")

    # Storing the metrics as a pandas dataframe so that comparisons can be made
    pd.DataFrame({"precision": [precision],
                  "recall": [recall],
                  "score_threshold": [SCORE_THRESHOLD],
                  "iou_threshold": [IOU_THRESHOLD]}).to_csv(out_metrics.path, index=False)

    # Saving per scene results and model
    pd.DataFrame(per_scene_results).to_csv(out_scene_results.path, index=False)

    cpu_state = {k: v.cpu() for k, v in model_fold0.state_dict().items()}  # loadable on CPU-only machines
    torch.save(cpu_state, out_model.path)
    out_model.metadata["file_type"] = ".pt"
    out_model.metadata["algo"] = "cnn_sliding_window"



#### Pipeline Component: Downloading The Metrics Data of The Model Currently in The Model Bucket ####
@dsl.component(
    packages_to_install=["google-cloud-storage", "pandas"],
    base_image="python:3.12.7-slim"
)

def download_metrics(project_id: str, bucket:str, 
                     # Metrics come from a folder in the model repo bucket
                     in_current_metrics: str, out_current_metrics: Output[Dataset]): 
    import logging
    import sys
    import pandas as pd
    from google.cloud import storage

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket)
    blob = bucket.blob(in_current_metrics)
    blob.download_to_filename(out_current_metrics.path)
    logging.info('Downloaded Metrics of Model Currently in Model Bucket!')




#### Pipeline Component: Comparing The Model from This CT Pipeline with The Model Currently in The Model Bucket ####
@dsl.component(
    packages_to_install=["pandas"],
    base_image="python:3.12.7-slim"
)

def compare_models(# ct_metrics are the metrics from the current run of the model. Current metrics are the metrics of the model currently in the model bucket
                   in_ct_metrics: Input[Dataset], in_current_metrics: Input[Dataset]) -> str:
    import logging
    import sys
    import pandas as pd

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Read in the metrics
    ct_metrics = pd.read_csv(in_ct_metrics.path)
    current_metrics = pd.read_csv(in_current_metrics.path)

    # We have precision and recall. The best metric to compare both is thus the F1-score, which balances both precision and recall
    f1_ct = ((2 * ct_metrics["precision"] * ct_metrics["recall"])/(ct_metrics["precision"] + ct_metrics["recall"]))[0]
    f1_current = ((2 * current_metrics["precision"] * current_metrics["recall"])/(current_metrics["precision"] + current_metrics["recall"]))[0]

    logging.info(f"CT Pipeline Model F1-Score: {f1_ct}")
    logging.info(f"Current Model in Bucket F1-Score: {f1_current}")


    # We then compare the f1 scores, and only upload the model if the ct pipeline's score is larger than the current one
    if f1_ct > f1_current:
        logging.info("CT Pipeline has a larger F1-score and thus performs better. Hence, this model will be uploaded and replace the current model in the bucket")
        return "CT"
    else:
        logging.info("Current model still has a larger F1-score than the model from the CT Pipeline. Thus, the CT Pipeline's model will NOT be uploaded, and the current model prevails")
        return "CURRENT"
    



#### Pipeline Component: Upload Model and Metrics to GCP Bucket Only if Trained Model is Better than Current Model ####
@dsl.component(
    packages_to_install=["google-cloud-storage", "pandas"],
    base_image="python:3.12.7-slim"
)
def upload_model_to_gcs(project_id: str, model_repo: str, model: Input[Model], 
                        ct_model_metrics: Input[Dataset], ct_model_scene_results: Input[Dataset]):
    '''upload model to gsc'''
    from google.cloud import storage   
    import logging 
    import sys
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)    
  
    # upload the model to GCS
    client = storage.Client(project=project_id)
    bucket = client.bucket(model_repo)
    mb_path = str(model.metadata["algo"]) + '/' + str(model.metadata["algo"]) + '_model' + str(model.metadata["file_type"])
    blob = bucket.blob(mb_path) 
    blob.upload_from_filename(model.path)       
    logging.info("Saved the model to the folder : " + mb_path + "in the bucket" + model_repo)

    # upload metrics to GCS
    metrics_path = "metrics/model_metrics.csv"
    blob_metrics = bucket.blob(metrics_path)
    blob_metrics.upload_from_filename(ct_model_metrics.path)
    logging.info("Saved the metrics to the folder : " + metrics_path + "in the bucket" + model_repo)

    scene_res_path = "metrics/scene_results.csv"
    blob_scene_res = bucket.blob(scene_res_path)
    blob_scene_res.upload_from_filename(ct_model_scene_results.path)
    logging.info("Saved the metrics to the folder : " + scene_res_path + "in the bucket" + model_repo)



#### Defining the Workflow of the Pipeline ####
# Define the workflow of the pipeline.
@kfp.dsl.pipeline(
    name="wheres-waldo-sliding-window-training-pipeline")
def pipeline(project_id: str, data_bucket: str, where_waldo_filename: str, 
             hey_waldo_filename:str, model_repo: str, current_metrics_filename: str):
    
    
    di_op = download_data(
        project_id = project_id, 
        bucket = data_bucket, 
        file_name = where_waldo_filename
    )

    prepro_op = preprocess_data(
        project_id = project_id,
        bucket=data_bucket, 
        # for the hey_waldo expansion, zip file from bucket
        hey_waldo_blob=hey_waldo_filename,
        in_manifest=di_op.outputs["dataset"], 
        # Zip file of images stored from previous component
        in_images=di_op.outputs["images"])
    

    training_cnn_job_run_op = train_cnn(
        in_patch_manifest = prepro_op.outputs["out_exp_manifest"], # Comes from expanded_patch_manifest
        # Comes from the fold assignments
        in_fold_assignment = prepro_op.outputs["out_fold_assignment"],
        # Comes from manifest_to_save
        in_scenes = di_op.outputs["dataset"], 
        # Zip file of images
        in_images = di_op.outputs["images"],
        # Zip file of patches
        in_patches = prepro_op.outputs["out_patches"])

    download_metrics_op = download_metrics(
        project_id=project_id,
        # Download the metrics of the model currently stored in the model repo
        bucket=model_repo,
        in_current_metrics=current_metrics_filename
    )

    comp_model_op = compare_models(
        in_ct_metrics=training_cnn_job_run_op.outputs["out_metrics"],
        in_current_metrics=download_metrics_op.outputs["out_current_metrics"]
        
    )
    # Only upload the model if the CT model is better
    with dsl.If(comp_model_op.output=="CT"):
        upload_model_op = upload_model_to_gcs(
            project_id = project_id,
            model_repo = model_repo,
            model = training_cnn_job_run_op.outputs["out_model"],
            ct_model_metrics=training_cnn_job_run_op.outputs["out_metrics"],
            ct_model_scene_results=training_cnn_job_run_op.outputs["out_scene_results"]
        ).after(comp_model_op)



# Function to compile pipeline
def compile_pipeline(target_file):
    compiler.Compiler().compile(pipeline_func=pipeline,
                                package_path=target_file)


# Function to parse the CLI arguments
def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_file', type=str, default="wheres-waldo-sliding-window-predictor.yaml", help="Pipeline Name")
    args = parser.parse_args()
    return vars(args)

# Only execute if this is the script being executed
if __name__ == '__main__':
    compile_pipeline(**parse_command_line_arguments())