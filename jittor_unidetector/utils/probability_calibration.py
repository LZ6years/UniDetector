import jittor as jt
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV

class ProbabilityCalibrator:
    """
    概率校准器
    用于改善检测置信度的可靠性
    """
    
    def __init__(self, method='isotonic', cv=5):
        self.method = method
        self.cv = cv
        self.calibrators = {}
        self.is_fitted = False
    
    def fit(self, scores, labels):
        """
        训练校准器
        Args:
            scores: 原始分数 [N, num_classes]
            labels: 真实标签 [N]
        """
        num_classes = scores.shape[1]
        
        for i in range(num_classes):
            # 二分类校准
            binary_labels = (labels == i).astype(np.float32)
            class_scores = scores[:, i]
            
            if self.method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
            else:
                calibrator = CalibratedClassifierCV(
                    method=self.method, cv=self.cv)
            
            # 训练校准器
            calibrator.fit(class_scores.reshape(-1, 1), binary_labels)
            self.calibrators[i] = calibrator
        
        self.is_fitted = True
    
    def calibrate(self, scores):
        """
        校准分数
        Args:
            scores: 原始分数 [N, num_classes]
        Returns:
            calibrated_scores: 校准后的分数 [N, num_classes]
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        calibrated_scores = np.zeros_like(scores)
        
        for i in range(scores.shape[1]):
            if i in self.calibrators:
                class_scores = scores[:, i]
                calibrated_scores[:, i] = self.calibrators[i].predict_proba(
                    class_scores.reshape(-1, 1))[:, 1]
        
        return calibrated_scores


class PriorProbabilityCalibrator:
    """
    基于先验概率的校准器
    使用历史检测结果的分布进行校准
    """
    
    def __init__(self, prior_path=None):
        self.prior_path = prior_path
        self.prior_probs = None
        self.is_fitted = False
    
    def fit_from_file(self, prior_path):
        """
        从文件加载先验概率
        Args:
            prior_path: 先验概率文件路径
        """
        if prior_path and jt.misc.exists(prior_path):
            self.prior_probs = np.load(prior_path)
            self.is_fitted = True
        else:
            raise FileNotFoundError(f"Prior probability file not found: {prior_path}")
    
    def fit_from_detections(self, detection_results):
        """
        从检测结果计算先验概率
        Args:
            detection_results: 检测结果列表
        """
        # 统计每个类别的检测次数
        class_counts = {}
        total_detections = 0
        
        for result in detection_results:
            for detection in result:
                class_id = detection['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_detections += 1
        
        # 计算先验概率
        num_classes = max(class_counts.keys()) + 1
        self.prior_probs = np.zeros(num_classes)
        
        for class_id, count in class_counts.items():
            self.prior_probs[class_id] = count / total_detections
        
        self.is_fitted = True
    
    def calibrate(self, scores, temperature=1.0):
        """
        使用先验概率校准分数
        Args:
            scores: 原始分数 [N, num_classes]
            temperature: 温度参数
        Returns:
            calibrated_scores: 校准后的分数 [N, num_classes]
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibration")
        
        # 应用温度缩放
        scaled_scores = scores / temperature
        
        # 应用先验概率
        calibrated_scores = scaled_scores * self.prior_probs
        
        # 归一化
        calibrated_scores = calibrated_scores / calibrated_scores.sum(axis=1, keepdims=True)
        
        return calibrated_scores


class EnsembleCalibrator:
    """
    集成校准器
    结合多种校准方法
    """
    
    def __init__(self, calibrators, weights=None):
        self.calibrators = calibrators
        self.weights = weights if weights is not None else [1.0] * len(calibrators)
        
    def fit(self, scores, labels, detection_results=None):
        """
        训练所有校准器
        Args:
            scores: 原始分数
            labels: 真实标签
            detection_results: 检测结果（用于先验概率）
        """
        for i, calibrator in enumerate(self.calibrators):
            if isinstance(calibrator, PriorProbabilityCalibrator):
                if detection_results is not None:
                    calibrator.fit_from_detections(detection_results)
                else:
                    raise ValueError("Detection results required for prior probability calibrator")
            else:
                calibrator.fit(scores, labels)
    
    def calibrate(self, scores, **kwargs):
        """
        集成校准
        Args:
            scores: 原始分数
            **kwargs: 其他参数
        Returns:
            calibrated_scores: 校准后的分数
        """
        calibrated_scores_list = []
        
        for calibrator in self.calibrators:
            calibrated_scores = calibrator.calibrate(scores, **kwargs)
            calibrated_scores_list.append(calibrated_scores)
        
        # 加权平均
        final_scores = np.zeros_like(scores)
        for i, (calibrated_scores, weight) in enumerate(zip(calibrated_scores_list, self.weights)):
            final_scores += weight * calibrated_scores
        
        return final_scores 