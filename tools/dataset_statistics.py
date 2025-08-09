#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Universal dataset statistics tool for COCO format datasets
Supports COCO, LVIS, Object365, and any other COCO-compatible datasets
"""

import json
import argparse
import os
from collections import defaultdict, Counter
import numpy as np

def load_annotation_file(annotation_path):
    """
    Load annotation file with error handling
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: Annotation file not found: {annotation_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in {annotation_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading {annotation_path}: {e}")
        return None

def analyze_dataset_statistics(data):
    """
    Analyze comprehensive statistics of a dataset
    """
    if not data:
        return None
    
    stats = {}
    
    # Basic counts
    stats['total_images'] = len(data.get('images', []))
    stats['total_annotations'] = len(data.get('annotations', []))
    stats['total_categories'] = len(data.get('categories', []))
    
    # Image statistics
    if data.get('images'):
        images = data['images']
        stats['image_stats'] = {
            'widths': [img.get('width', 0) for img in images],
            'heights': [img.get('height', 0) for img in images],
            'file_names': [img.get('file_name', '') for img in images],
            'ids': [img.get('id', 0) for img in images]
        }
        
        # Image size statistics
        widths = stats['image_stats']['widths']
        heights = stats['image_stats']['heights']
        if widths and heights:
            stats['image_size_stats'] = {
                'width': {
                    'min': min(widths),
                    'max': max(widths),
                    'mean': np.mean(widths),
                    'median': np.median(widths)
                },
                'height': {
                    'min': min(heights),
                    'max': max(heights),
                    'mean': np.mean(heights),
                    'median': np.median(heights)
                }
            }
    
    # Category statistics
    if data.get('categories'):
        categories = data['categories']
        stats['category_stats'] = {
            'ids': [cat.get('id', 0) for cat in categories],
            'names': [cat.get('name', '') for cat in categories],
            'supercategories': [cat.get('supercategory', '') for cat in categories]
        }
    
    # Annotation statistics
    if data.get('annotations'):
        annotations = data['annotations']
        
        # Category distribution
        category_counts = Counter([ann.get('category_id', 0) for ann in annotations])
        stats['category_distribution'] = dict(category_counts)
        
        # Image annotation counts
        image_annotation_counts = Counter([ann.get('image_id', 0) for ann in annotations])
        stats['image_annotation_counts'] = dict(image_annotation_counts)
        
        # Annotation type analysis
        annotation_types = Counter([ann.get('iscrowd', 0) for ann in annotations])
        stats['annotation_types'] = {
            'individual': annotation_types.get(0, 0),
            'crowd': annotation_types.get(1, 0)
        }
        
        # Bounding box statistics
        bbox_areas = []
        bbox_widths = []
        bbox_heights = []
        
        for ann in annotations:
            bbox = ann.get('bbox', [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                area = w * h
                bbox_areas.append(area)
                bbox_widths.append(w)
                bbox_heights.append(h)
        
        if bbox_areas:
            stats['bbox_stats'] = {
                'area': {
                    'min': min(bbox_areas),
                    'max': max(bbox_areas),
                    'mean': np.mean(bbox_areas),
                    'median': np.median(bbox_areas)
                },
                'width': {
                    'min': min(bbox_widths),
                    'max': max(bbox_widths),
                    'mean': np.mean(bbox_widths),
                    'median': np.median(bbox_widths)
                },
                'height': {
                    'min': min(bbox_heights),
                    'max': max(bbox_heights),
                    'mean': np.mean(bbox_heights),
                    'median': np.median(bbox_heights)
                }
            }
    
    return stats

def print_statistics(stats, dataset_name="Dataset"):
    """
    Print formatted statistics
    """
    if not stats:
        print("No statistics available")
        return
    
    print(f"\n{'='*60}")
    print(f"📊 {dataset_name} Statistics")
    print(f"{'='*60}")
    
    # Basic counts
    print(f"\n📈 Basic Counts:")
    print(f"  • Total Images: {stats['total_images']:,}")
    print(f"  • Total Annotations: {stats['total_annotations']:,}")
    print(f"  • Total Categories: {stats['total_categories']:,}")
    print(f"  • Average Annotations per Image: {stats['total_annotations']/stats['total_images']:.2f}")
    
    # Image statistics
    if 'image_size_stats' in stats:
        print(f"\n🖼️  Image Size Statistics:")
        width_stats = stats['image_size_stats']['width']
        height_stats = stats['image_size_stats']['height']
        print(f"  • Width:  {width_stats['min']} - {width_stats['max']} (mean: {width_stats['mean']:.1f}, median: {width_stats['median']:.1f})")
        print(f"  • Height: {height_stats['min']} - {height_stats['max']} (mean: {height_stats['mean']:.1f}, median: {height_stats['median']:.1f})")
    
    # Category distribution
    if 'category_distribution' in stats:
        print(f"\n🏷️  Category Distribution:")
        category_dist = stats['category_distribution']
        top_categories = sorted(category_dist.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"  • Top 10 categories by annotation count:")
        for cat_id, count in top_categories:
            print(f"    - Category {cat_id}: {count:,} annotations")
    
    # Annotation distribution
    if 'image_annotation_counts' in stats:
        img_ann_counts = stats['image_annotation_counts']
        if img_ann_counts:
            counts_list = list(img_ann_counts.values())
            print(f"\n📋 Annotation Distribution per Image:")
            print(f"  • Min annotations per image: {min(counts_list)}")
            print(f"  • Max annotations per image: {max(counts_list)}")
            print(f"  • Mean annotations per image: {np.mean(counts_list):.2f}")
            print(f"  • Median annotations per image: {np.median(counts_list):.2f}")
    
    # Bounding box statistics
    if 'bbox_stats' in stats:
        print(f"\n📦 Bounding Box Statistics:")
        bbox_stats = stats['bbox_stats']
        area_stats = bbox_stats['area']
        width_stats = bbox_stats['width']
        height_stats = bbox_stats['height']
        print(f"  • Area:     {area_stats['min']:.1f} - {area_stats['max']:.1f} (mean: {area_stats['mean']:.1f})")
        print(f"  • Width:    {width_stats['min']:.1f} - {width_stats['max']:.1f} (mean: {width_stats['mean']:.1f})")
        print(f"  • Height:   {height_stats['min']:.1f} - {height_stats['max']:.1f} (mean: {height_stats['mean']:.1f})")
    
    # Annotation types
    if 'annotation_types' in stats:
        print(f"\n👥 Annotation Types:")
        ann_types = stats['annotation_types']
        print(f"  • Individual annotations: {ann_types['individual']:,}")
        print(f"  • Crowd annotations: {ann_types['crowd']:,}")

def save_statistics(stats, output_file):
    """
    Save statistics to JSON file
    """
    if not stats:
        return
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Statistics saved to: {output_file}")
    except Exception as e:
        print(f"Error saving statistics: {e}")

def main():
    parser = argparse.ArgumentParser(description='Universal dataset statistics tool')
    parser.add_argument('--annotation', '-a', type=str, required=True,
                       help='Path to annotation file (JSON)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file to save statistics (optional)')
    parser.add_argument('--name', '-n', type=str, default='Dataset',
                       help='Dataset name for display')
    
    args = parser.parse_args()
    
    # Load annotation file
    print(f"Loading annotation file: {args.annotation}")
    data = load_annotation_file(args.annotation)
    
    if not data:
        print("Failed to load annotation file")
        return
    
    # Analyze statistics
    print("Analyzing dataset statistics...")
    stats = analyze_dataset_statistics(data)
    
    # Print statistics
    print_statistics(stats, args.name)
    
    # Save statistics if output file specified
    if args.output:
        save_statistics(stats, args.output)

if __name__ == '__main__':
    main() 