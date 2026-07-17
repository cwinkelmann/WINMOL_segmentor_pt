#!/usr/bin/env python
"""Build a SITE-HELD-OUT BAMFORESTS dataset for a generalization benchmark.

BAMFORESTS has three forest sites: **Tretzendorf** and **Stadtwald** (in the train/eval/
TestSet2 splits) and **Hain** (TestSet1 only). To measure whether a model *generalizes to a
new location* — rather than memorizing the training sites — the held-out test must be a site
the model never saw. This builder maps the official COCO splits so that:

  train        <- train2023   (Tretzendorf + Stadtwald)
  val          <- eval2023    (Tretzendorf + Stadtwald)   # model selection, in-distribution
  test         <- TestSet1    (Hain ONLY)                 # HELD-OUT location -> generalization
  test_insite  <- TestSet2    (Tretzendorf + Stadtwald)   # optional in-distribution test;
                                                            # (test_insite F1 - test F1) = the
                                                            # cross-site generalization gap

train and val share sites (standard: select on the training distribution); **test is a
disjoint location**. The builder asserts train/val sites are disjoint from the test site and
refuses to proceed on any leak.

Each split is materialized to `<dst>/<split>/{train,mask}/` in loader format via the shared
`coco_to_dataset()` (RGB jpeg + binary gif, polygons only). A `SITES.md` manifest records the
site→split mapping and the leakage-check result.

Usage:
  python scripts/build_bamforests_sitesplit.py --coco-root <coco1024> --dst <out> \
    [--limit-train 4000 --limit-val 1000 --limit-test 1600 --limit-insite 0] [--seed 1]
  # limit 0 (default) = all annotated images in that split.
"""
import argparse
import json
import os
import re

from coco_to_dataset import coco_to_dataset

# split -> (annotation json basename, images subdir under --coco-root)
SPLITS = {
    "train":       ("instances_tree_train2023.json",    "train2023"),
    "val":         ("instances_tree_eval2023.json",     "val2023"),
    "test":        ("instances_tree_TestSet12023.json", "test2023/Test-Set-1"),
    "test_insite": ("instances_tree_TestSet22023.json", "test2023/Test-Set-2"),
}
TRAINVAL = ("train", "val")           # share sites; must be disjoint from `test`


def _site(fn):
    m = re.match(r"^([A-Za-z]+)", os.path.basename(fn))
    return m.group(1) if m else "?"


def _sites_of(coco_json):
    coco = json.load(open(coco_json))
    annotated = {a["image_id"] for a in coco["annotations"]}
    return {_site(im["file_name"]) for im in coco["images"] if im["id"] in annotated}


def _annotated_count(coco_json):
    coco = json.load(open(coco_json))
    return len({a["image_id"] for a in coco["annotations"]})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coco-root", required=True, help="BAMFORESTS coco1024 dir (annotations/, "
                   "train2023/, val2023/, test2023/)")
    p.add_argument("--dst", required=True, help="output dir; splits written as <dst>/<split>")
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    p.add_argument("--limit-insite", type=int, default=0, help="0 = all; --no-insite to skip")
    p.add_argument("--no-insite", action="store_true",
                   help="skip the TestSet2 in-distribution test split")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    ann = os.path.join(a.coco_root, "annotations")
    limits = {"train": a.limit_train, "val": a.limit_val, "test": a.limit_test,
              "test_insite": a.limit_insite}
    splits = {k: v for k, v in SPLITS.items() if not (k == "test_insite" and a.no_insite)}

    # --- leakage guard: test site(s) must be disjoint from train+val sites ---
    sites = {k: _sites_of(os.path.join(ann, j)) for k, (j, _) in splits.items()}
    trainval_sites = set().union(*(sites[k] for k in TRAINVAL))
    leak = trainval_sites & sites["test"]
    if leak:
        raise SystemExit(f"SITE LEAK: test shares site(s) {sorted(leak)} with train/val "
                         f"({sorted(trainval_sites)}); test must be a held-out location.")
    print(f"[sitesplit] train/val sites={sorted(trainval_sites)} | "
          f"test site(s)={sorted(sites['test'])} -> disjoint OK")

    manifest = [f"# BAMFORESTS site-held-out split\n",
                f"Source: `{a.coco_root}` | seed {a.seed}\n",
                "| split | source | sites | images |", "|---|---|---|---:|"]
    for split, (jbase, imgsub) in splits.items():
        coco_json = os.path.join(ann, jbase)
        n_all = _annotated_count(coco_json)
        limit = limits[split] or n_all
        limit = min(limit, n_all)
        dst = os.path.join(a.dst, split)
        print(f"[sitesplit] {split}: {jbase} ({sorted(sites[split])}) -> {dst} "
              f"(limit {limit}/{n_all})", flush=True)
        n = coco_to_dataset(coco_json, os.path.join(a.coco_root, imgsub), dst,
                            limit=limit, seed=a.seed)
        manifest.append(f"| {split} | {jbase} | {', '.join(sorted(sites[split]))} | {n} |")
    manifest += ["",
                 f"**Held-out test location:** {', '.join(sorted(sites['test']))} "
                 f"(disjoint from train/val sites {', '.join(sorted(trainval_sites))}).",
                 "`test` measures cross-site generalization; `test_insite` (if present) is an "
                 "in-distribution test — the gap between them is the generalization penalty."]
    os.makedirs(a.dst, exist_ok=True)
    with open(os.path.join(a.dst, "SITES.md"), "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"[sitesplit] done -> {a.dst} (+ SITES.md)")


if __name__ == "__main__":
    main()
