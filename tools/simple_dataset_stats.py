#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple dataset statistics tool - only shows Images, Annotations, Categories
"""

import json
import argparse
import os

def get_basic_stats(annotation_path):
    """
    Get basic statistics: Images, Annotations, Categories
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        images = len(data.get('images', []))
        annotations = len(data.get('annotations', []))
        categories = len(data.get('categories', []))
        
        return images, annotations, categories
    except Exception as e:
        print(f"Error reading {annotation_path}: {e}")
        return 0, 0, 0

def main():
    # List of annotation files to analyze
    annotation_files = [
        # "/root/autodl-tmp/datasets/object365/annotations/zhiyuan_objv2_train_patch0-5.1@2.0.json",
        # "/root/autodl-tmp/datasets/lvis_v1.0/annotations/lvis_v1_train.1@1.0.json",
        "/root/autodl-tmp/datasets/lvis_v1.0/annotations/lvis_v1_val.1@4.0.json",
        # "/root/autodl-tmp/datasets/coco/annotations/instances_train2017.1@5.0.json",
        # "/root/autodl-tmp/datasets/coco/annotations/instances_val2017.1@25.0.json",
    ]
    
    # Dataset names for display
    dataset_names = [
        # "Object365 Patch0-5 (1%)",
        # "LVIS Train (1%)",
        "LVIS Val (4%)",
        # "COCO Train (5%)",
        # "COCO Val (25%)",
    ]
    
    print("=" * 80)
    print(f"{'Dataset':<25} {'Images':<10} {'Annotations':<12} {'Categories':<10}")
    print("=" * 80)
    
    for i, (file_path, name) in enumerate(zip(annotation_files, dataset_names)):
        if os.path.exists(file_path):
            images, annotations, categories = get_basic_stats(file_path)
            print(f"{name:<25} {images:<10,} {annotations:<12,} {categories:<10,}")
        else:
            print(f"{name:<25} {'File not found':<10} {'N/A':<12} {'N/A':<10}")
    
    print("=" * 80)

if __name__ == '__main__':
    main() 