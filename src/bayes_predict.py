#encoding=utf-8
import math
import pickle
from collections import defaultdict

from src.bayes import extract_words_with_weight

# 必须与保存时的定义完全一致
def default_class_stats():
    return {
        "docs_count": 0,
        "weighted_word_freq": defaultdict(float),
        "total_weighted_words": 0.0,
        "doc_contain_word": defaultdict(int)
    }
class Bayes_Predict:
    """
        实际使用时，使用该实体
    """

    def __init__(self):
        self.model = self.load_model("../src/nb_model.pkl")

    def load_model(self,model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def classify_with_weight(self,text):
        """带权重的分类函数"""
        model=self.model
        words_with_weight = extract_words_with_weight(text)
        max_log_prob = -float("inf")
        best_class = None

        for cls in model["classes"]:
            log_prob = math.log(model["prior_prob"][cls])
            # 累加带权概率
            for word, weight in words_with_weight:
                # 获取基础条件概率
                base_prob = model["conditional_prob"][cls].get(
                    word,
                    1 / (model["M_a"] + model["class_stats"][cls]["total_weighted_words"])
                )
                # 应用权重：P(W_k|C_i) × L_j
                weighted_prob = base_prob * weight
                log_prob += math.log(weighted_prob)

            if log_prob > max_log_prob:
                max_log_prob = log_prob
                best_class = cls

        return best_class