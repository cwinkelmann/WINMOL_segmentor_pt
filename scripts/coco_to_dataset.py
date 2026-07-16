#!/usr/bin/env python
"""Convert a COCO instance-segmentation split into the loader-ready train{N}.jpeg /
mask{N}.gif format, sampling a fixed number of annotated images.

For each sampled image: the source image is converted to RGB and saved as
``<dst>/train/train{i}.jpeg``; a **binary** mask (union of all category polygons -> 255)
is rasterized and saved as ``<dst>/mask/mask{i}.gif``. Segmentations must be polygons
(rasterized with PIL) — RLE is not supported (raises). By default only images with at
least one annotation are sampled (so masks are non-empty). Deterministic given --seed.

Usage:
  python scripts/coco_to_dataset.py \
    --coco-json .../annotations/instances_tree_train2023.json \
    --images-dir .../train2023 --dst .../100_images/train --limit 100 --seed 1
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw


def coco_to_dataset(coco_json, images_dir, dst, limit=100, seed=1, require_annotations=True):
    coco = json.load(open(coco_json))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2size = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    ids = [i for i in id2file if anns_by_img.get(i)] if require_annotations else list(id2file)
    ids.sort()                                   # deterministic order before shuffle
    random.Random(seed).shuffle(ids)
    chosen = ids[:limit]
    if len(chosen) < limit:
        raise ValueError(f"only {len(chosen)} eligible images (< limit {limit}) in {coco_json}")

    img_dst = os.path.join(dst, "train")
    mask_dst = os.path.join(dst, "mask")
    os.makedirs(img_dst, exist_ok=True)
    os.makedirs(mask_dst, exist_ok=True)

    for i, image_id in enumerate(chosen, start=1):
        Image.open(os.path.join(images_dir, id2file[image_id])).convert("RGB").save(
            os.path.join(img_dst, f"train{i}.jpeg"))
        w, h = id2size[image_id]
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)
        for a in anns_by_img.get(image_id, []):
            seg = a["segmentation"]
            if isinstance(seg, dict):
                raise ValueError(f"RLE segmentation not supported (image_id {image_id})")
            for poly in seg:                     # poly = [x1,y1,x2,y2,...]
                pts = list(zip(poly[0::2], poly[1::2]))
                if len(pts) >= 3:
                    draw.polygon(pts, fill=255)
        mask.save(os.path.join(mask_dst, f"mask{i}.gif"))
    return len(chosen)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-json", required=True)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--allow-empty", action="store_true",
                   help="also sample images with no annotations (empty masks)")
    a = p.parse_args()
    n = coco_to_dataset(a.coco_json, a.images_dir, a.dst, a.limit, a.seed,
                        require_annotations=not a.allow_empty)
    print(f"wrote {n} pairs to {a.dst}")


if __name__ == "__main__":
    main()
