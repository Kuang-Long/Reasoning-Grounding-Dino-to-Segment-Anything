import numpy as np
import cv2
import os
import csv

def calculate_giou(output_mask, gt_valid_mask, gt_ignore_mask=None):
    """
    計算 Generalized Intersection over Union (gIoU)
    Args:
        output_mask (numpy.ndarray): 模型輸出的遮罩，值為 0 或 1
        gt_valid_mask (numpy.ndarray): Ground Truth 的有效區域，值為 0 或 1
        gt_ignore_mask (numpy.ndarray): Ground Truth 的忽略區域，值為 0 或 1，默認為 None
    Returns:
        float: gIoU 分數
    """
    intersection = np.logical_and(output_mask == 1, gt_valid_mask == 1)
    union = np.logical_or(output_mask == 1, gt_valid_mask == 1)
    
    if gt_ignore_mask is not None:
        ignore_area = gt_ignore_mask == 1
        union = np.logical_and(union, ~ignore_area)

    intersection_area = np.sum(intersection)
    union_area = np.sum(union)

    return intersection_area / union_area if union_area > 0 else 0.0

def calculate_ciou(output_mask, gt_valid_mask, gt_ignore_mask=None):
    """
    計算 Complete Intersection over Union (cIoU)
    Args:
        output_mask (numpy.ndarray): 模型輸出的遮罩，值為 0 或 1
        gt_valid_mask (numpy.ndarray): Ground Truth 的有效區域，值為 0 或 1
        gt_ignore_mask (numpy.ndarray): Ground Truth 的忽略區域，值為 0 或 1，默認為 None
    Returns:
        float: cIoU 分數
    """
    intersection = np.logical_and(output_mask == 1, gt_valid_mask == 1)
    valid_area = np.sum(gt_valid_mask == 1)
    
    if gt_ignore_mask is not None:
        ignore_area = gt_ignore_mask == 1
        valid_area = np.sum(np.logical_and(gt_valid_mask == 1, ~ignore_area))
    
    intersection_area = np.sum(intersection)

    return intersection_area / valid_area if valid_area > 0 else 0.0

def evaluate_masks(output_dir, gt_dir, gt_ignore_dir=None):
    """
    遍歷所有測試遮罩文件，根據檔案名稱的前半部分匹配對應關係，計算平均 gIoU 和 cIoU。
    Args:
        output_dir (str): 模型輸出遮罩的目錄
        gt_dir (str): Ground Truth 遮罩的目錄
        gt_ignore_dir (str): Ground Truth 忽略區域的目錄，默認為 None
    Returns:
        float, float: 平均 gIoU 和 cIoU 分數
    """
    giou_scores = []
    ciou_scores = []
    results = []  # 用于存储每个文件的评估结果

    # 提取模型輸出、Ground Truth 和忽略遮罩的基礎檔案名稱
    output_files = {os.path.splitext(f)[0].split('_mask_0')[0]: f for f in os.listdir(output_dir)}
    gt_files = {os.path.splitext(f)[0].split('_valid_mask')[0]: f for f in os.listdir(gt_dir)}
    ignore_files = {}
    if gt_ignore_dir:
        ignore_files = {os.path.splitext(f)[0].split('_ignore_mask')[0]: f for f in os.listdir(gt_ignore_dir)}

    # 找出名稱匹配的檔案
    matching_files = set(output_files.keys()).intersection(set(gt_files.keys()))
    if gt_ignore_dir:
        matching_files = matching_files.intersection(set(ignore_files.keys()))

    if not matching_files:
        print("No matching files found based on base names.")
        return 0.0, 0.0

    for base_name in sorted(matching_files):
        print(f"Evaluating {base_name}.jpg...")
        # 確保對應檔案存在於輸出、Ground Truth 和忽略遮罩資料夾
        output_file = output_files[base_name]
        gt_file = gt_files[base_name]
        ignore_file = ignore_files.get(base_name) if gt_ignore_dir else None

        # 讀取輸出遮罩和 Ground Truth 遮罩
        output_mask = cv2.imread(os.path.join(output_dir, output_file), cv2.IMREAD_GRAYSCALE)
        gt_valid_mask = cv2.imread(os.path.join(gt_dir, gt_file), cv2.IMREAD_GRAYSCALE)

        # 如果忽略區域存在，讀取忽略區域
        gt_ignore_mask = None
        if ignore_file:
            gt_ignore_mask = cv2.imread(os.path.join(gt_ignore_dir, ignore_file), cv2.IMREAD_GRAYSCALE)

        # 確保所有遮罩轉為二值化
        output_mask = (output_mask > 0).astype(np.uint8)
        gt_valid_mask = (gt_valid_mask > 0).astype(np.uint8)
        if gt_ignore_mask is not None:
            gt_ignore_mask = (gt_ignore_mask > 0).astype(np.uint8)

        # 計算 gIoU 和 cIoU
        giou = calculate_giou(output_mask, gt_valid_mask, gt_ignore_mask)
        ciou = calculate_ciou(output_mask, gt_valid_mask, gt_ignore_mask)
        giou_scores.append(giou)
        ciou_scores.append(ciou)
        print(f"gIoU: {giou:.4f}, cIoU: {ciou:.4f}")
    
        results.append([base_name+".jpg", giou, ciou])

    # 計算平均 gIoU 和 cIoU
    avg_giou = np.mean(giou_scores) if giou_scores else 0.0
    avg_ciou = np.mean(ciou_scores) if ciou_scores else 0.0

    return avg_giou, avg_ciou , results



if __name__ == "__main__":
    # 設定資料夾路徑
    output_dir = "./LISA8bit_baseline/mask"  # 模型輸出的遮罩目錄
    gt_ignore_dir = "./ReasonSeg_GT/ignore_masks"  # Ground Truth 忽略遮罩目錄，可為 None
    gt_valid_dir = "./ReasonSeg_GT/valid_masks"  # Ground Truth 有效遮罩目錄


    # 執行評估
    avg_giou, avg_ciou, results = evaluate_masks(output_dir, gt_valid_dir, gt_ignore_dir)

    # 輸出結果
    print(f"Average gIoU: {avg_giou:.4f}")
    print(f"Average cIoU: {avg_ciou:.4f}")

    with open('baseline_evaluation_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'gIoU', 'cIoU'])  # 写入表头
        writer.writerows(results)  # 写入每个文件的评估结果
        writer.writerow(['Average', avg_giou, avg_ciou])

