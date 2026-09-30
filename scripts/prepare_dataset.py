"""
Dataset preparation, validation, and identity-grouping execution script.
Constructs verified identity groupings and prepares reproducible train/val/test splits.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.data.identity_index import CelebAIdentityIndexer
from modules.masking.synthetic_masks import MaskGenerator


def run_dataset_preparation(args):
    print("=" * 65)
    print("PHASE 1: DATASET VALIDATION & IDENTITY GROUPING")
    print("=" * 65)
    
    indexer = CelebAIdentityIndexer(
        identity_file=args.identity_file,
        mapping_file=args.mapping_file,
        output_index_file=args.output_index,
        downloads_dir=args.downloads_dir,
        random_seed=args.seed
    )
    
    index_data = indexer.load_or_build_index(force_rebuild=args.force)
    meta = index_data["metadata"]
    splits = index_data["splits"]
    
    print("\n--- Identity Grouping Statistics ---")
    print(f"Total multi-image identities (N >= 2) : {meta['total_identities_multi']}")
    print(f"Total images belonging to valid groups : {meta['total_images_in_groups']}")
    print(f"Train split identities                  : {meta['split_counts']['train']}")
    print(f"Validation split identities             : {meta['split_counts']['val']}")
    print(f"Test split identities                   : {meta['split_counts']['test']}")
    
    # Verify no split leakage
    train_set = set(splits["train"])
    val_set = set(splits["val"])
    test_set = set(splits["test"])
    
    assert len(train_set.intersection(val_set)) == 0, "DATA LEAKAGE: Train and Val overlap!"
    assert len(train_set.intersection(test_set)) == 0, "DATA LEAKAGE: Train and Test overlap!"
    assert len(val_set.intersection(test_set)) == 0, "DATA LEAKAGE: Val and Test overlap!"
    print("\n[Audit PASS] Zero identity leakage across train, val, and test partitions.")
    
    # Audit local image files
    available_images = []
    # Check unpacked directory
    unpacked_dir = os.path.join(args.downloads_dir, "CelebAMask-HQ", "CelebA-HQ-img")
    if os.path.exists(unpacked_dir):
        files = [f for f in os.listdir(unpacked_dir) if f.endswith(".jpg")]
        available_images.extend(files)
        print(f"[Audit] Found {len(files)} unpacked images in {unpacked_dir}")
        
    audit_report = {
        "dataset_validation": "SUCCESS",
        "metadata_index": meta,
        "leakage_test": "PASSED (0 intersections)",
        "unpacked_sample_images_available": len(available_images),
    }
    
    report_file = "data/metadata/dataset_audit_report.json"
    with open(report_file, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"[Audit] Wrote audit report to {report_file}")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CelebA-HQ Identity Grouping & Preparation")
    parser.add_argument("--identity-file", type=str, default="data/metadata/identity_CelebA.txt")
    parser.add_argument("--mapping-file", type=str, default="data/metadata/CelebA-HQ-to-CelebA-mapping.txt")
    parser.add_argument("--output-index", type=str, default="data/metadata/identity_index.json")
    parser.add_argument("--downloads-dir", type=str, default=r"C:\Users\sasan\Downloads")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Force re-indexing")
    
    args = parser.parse_args()
    run_dataset_preparation(args)
