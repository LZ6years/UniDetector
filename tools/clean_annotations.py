#!/usr/bin/env python3
"""
清理annotations文件，移除对应图像不存在的annotations
支持COCO和Objects365数据集
"""

import json
import os
import argparse
from tqdm import tqdm
from collections import defaultdict

def check_image_exists(image_path):
    """检查图像文件是否存在"""
    return os.path.exists(image_path)

def find_image_in_objects365(img_dir, file_name):
    """
    在Objects365数据集中查找图像文件
    Objects365的图像分布在patch1-patch5目录中
    """
    # 提取文件名（去掉路径前缀）
    if '/' in file_name:
        file_name = file_name.split('/')[-1]
    
    # 尝试不同的patch目录
    for patch_num in range(1, 6):
        patch_path = os.path.join(img_dir, f"patch{patch_num}", file_name)
        if os.path.exists(patch_path):
            return True
    return False

def clean_annotations(ann_file, img_dir, output_file=None, dataset_type='coco'):
    """
    清理annotations文件
    
    Args:
        ann_file: annotations文件路径
        img_dir: 图像目录路径
        output_file: 输出文件路径，如果为None则覆盖原文件
        dataset_type: 数据集类型 ('coco' 或 'objects365')
    """
    print(f"正在处理: {ann_file}")
    print(f"数据集类型: {dataset_type}")
    
    # 读取annotations文件
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    print(f"原始数据统计:")
    print(f"  图像数量: {len(data['images'])}")
    print(f"  标注数量: {len(data['annotations'])}")
    print(f"  类别数量: {len(data['categories'])}")
    
    # 检查图像文件是否存在
    print("正在检查图像文件...")
    valid_image_ids = set()
    missing_images = []
    
    for img_info in tqdm(data['images'], desc="检查图像文件"):
        if dataset_type == 'objects365':
            # Objects365的特殊处理
            exists = find_image_in_objects365(img_dir, img_info['file_name'])
        else:
            # COCO的标准处理
            img_path = os.path.join(img_dir, img_info['file_name'])
            exists = check_image_exists(img_path)
        
        if exists:
            valid_image_ids.add(img_info['id'])
        else:
            missing_images.append(img_info['file_name'])
    
    print(f"有效图像数量: {len(valid_image_ids)}")
    print(f"缺失图像数量: {len(missing_images)}")
    
    if missing_images:
        print("缺失的图像文件:")
        for img in missing_images[:10]:  # 只显示前10个
            print(f"  - {img}")
        if len(missing_images) > 10:
            print(f"  ... 还有 {len(missing_images) - 10} 个文件")
    
    # 过滤annotations
    print("正在过滤annotations...")
    valid_annotations = []
    removed_annotations = 0
    
    for ann in tqdm(data['annotations'], desc="过滤annotations"):
        if ann['image_id'] in valid_image_ids:
            valid_annotations.append(ann)
        else:
            removed_annotations += 1
    
    print(f"保留的annotations: {len(valid_annotations)}")
    print(f"移除的annotations: {removed_annotations}")
    
    # 过滤images
    print("正在过滤images...")
    valid_images = []
    for img_info in data['images']:
        if img_info['id'] in valid_image_ids:
            valid_images.append(img_info)
    
    print(f"保留的images: {len(valid_images)}")
    
    # 创建新的数据
    cleaned_data = {
        'info': data.get('info', {}),
        'licenses': data.get('licenses', []),
        'categories': data['categories'],
        'images': valid_images,
        'annotations': valid_annotations
    }
    
    # 保存结果
    if output_file is None:
        output_file = ann_file
    
    print(f"正在保存到: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print("✅ 清理完成!")
    print(f"最终统计:")
    print(f"  图像数量: {len(cleaned_data['images'])}")
    print(f"  标注数量: {len(cleaned_data['annotations'])}")
    print(f"  类别数量: {len(cleaned_data['categories'])}")
    
    return cleaned_data

def main():
    parser = argparse.ArgumentParser(description='清理COCO/Objects365 annotations文件')
    parser.add_argument('--ann-file', type=str, required=True,
                       help='annotations文件路径')
    parser.add_argument('--img-dir', type=str, required=True,
                       help='图像目录路径')
    parser.add_argument('--output', type=str, default=None,
                       help='输出文件路径，默认覆盖原文件')
    parser.add_argument('--dataset-type', type=str, default='auto',
                       choices=['coco', 'objects365', 'auto'],
                       help='数据集类型，auto会自动检测')
    
    args = parser.parse_args()
    
    # 自动检测数据集类型
    if args.dataset_type == 'auto':
        if 'object365' in args.ann_file.lower() or 'objv2' in args.ann_file.lower():
            args.dataset_type = 'objects365'
        else:
            args.dataset_type = 'coco'
    
    # 检查输入文件是否存在
    if not os.path.exists(args.ann_file):
        print(f"❌ annotations文件不存在: {args.ann_file}")
        return
    
    if not os.path.exists(args.img_dir):
        print(f"❌ 图像目录不存在: {args.img_dir}")
        return
    
    # 执行清理
    clean_annotations(args.ann_file, args.img_dir, args.output, args.dataset_type)

if __name__ == '__main__':
    main() 