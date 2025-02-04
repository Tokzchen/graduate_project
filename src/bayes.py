# 定义标签分组及权重
import math
from collections import defaultdict

import jieba

TAG_GROUPS = {
    "group1": {
        "tags": {"title", "description", "keyword", "h1"},
        "weight": 2.0
    },
    "group2": {
        "tags": {"h2", "h3"},
        "weight": 1.5
    },
    "group3": {
        "tags": {"h4", "h5", "strong"},
        "weight": 1.2
    },
    "group4": {
        "tags": {"p", "td", "i"},
        "weight": 1.0
    },
    "group5": {
        "tags": None,  # 其他标签
        "weight": 0.2
    }
}

# 快速查询标签对应的权重
TAG_WEIGHT_MAP = {}
for group in TAG_GROUPS.values():
    if group["tags"]:
        for tag in group["tags"]:
            TAG_WEIGHT_MAP[tag] = group["weight"]

from bs4 import BeautifulSoup

stopwords = set(["的", "了", "在", "是", "和", ...])

def extract_words_with_weight(html_content):
    """解析HTML，提取词及其所在标签的权重"""
    soup = BeautifulSoup(html_content, "html.parser")
    words_with_weight = []

    # 遍历所有标签
    for tag in soup.find_all(True):
        tag_name = tag.name.lower()
        # 确定标签权重
        weight = TAG_WEIGHT_MAP.get(tag_name, TAG_GROUPS["group5"]["weight"])
        # 提取文本并分词
        text = tag.get_text()
        words = [word for word in jieba.lcut(text) if word not in stopwords]
        # 记录词及其权重
        words_with_weight.extend([(word, weight) for word in words])

    return words_with_weight


# ----------------------------
# 修改后的训练统计逻辑
# ----------------------------
class_stats = defaultdict(lambda: {
    "docs_count": 0,
    "weighted_word_freq": defaultdict(float),  # 带权词频
    "total_weighted_words": 0.0,  # 带权总词数
    "doc_contain_word": defaultdict(int)  # 包含某词的文档数（用于idf）
})

all_features = set() # 全局不重复的特征词集合

import csv
from collections import defaultdict
import math
import jieba
import random
import pickle


# ----------------------------
# Step 1: 读取数据并按类别划分
# ----------------------------
def load_and_split_data(csv_path, train_per_class=1000, test_per_class=200):
    # 按类别存储所有样本
    class_data = defaultdict(list)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            category, html_content = row
            class_data[category].append(html_content)

    # 划分训练集和测试集（每个类别取前1000训练，后200测试）
    train_data = []
    test_data = []
    for cls, contents in class_data.items():
        if len(contents) < 1200:
            raise ValueError(f"类别 {cls} 的样本不足1200条")
        train_data.extend([(cls, html) for html in contents[:train_per_class]])
        test_data.extend([(cls, html) for html in contents[train_per_class:train_per_class + test_per_class]])

    return train_data, test_data


# 示例调用
train_data, test_data = load_and_split_data("data.csv")

for cls, html in train_data:
    # 提取词及其权重
    words_with_weight = extract_words_with_weight(html)
    unique_words = set(word for word, _ in words_with_weight)

    # 更新类别统计
    stats = class_stats[cls]
    stats["docs_count"] += 1
    # 更新各个类每个特征词的词频统计
    for word in unique_words:
        stats["doc_contain_word"][word] += 1

    # 更新各类带权词频和总词数
    for word, weight in words_with_weight:
        stats["weighted_word_freq"][word] += weight
        stats["total_weighted_words"] += weight

    # 更新全局特征词
    all_features.update(word for word, _ in words_with_weight)

# ----------------------------
# 计算条件概率（带权）以及先验概率
# ----------------------------
prior_prob = {}
total_train_docs = len(train_data)
for cls in class_stats:
    prior_prob[cls] = class_stats[cls]["docs_count"] / total_train_docs

M_a=len(all_features) # 全局不重复特征词数
conditional_prob = defaultdict(dict)
for cls in class_stats:
    stats = class_stats[cls]
    M_i = stats["total_weighted_words"]  # 带权总词数

    for word in all_features:
        if word in stats["weighted_word_freq"]:
            # 带权词频 = Σ(词频 × 权重)
            weighted_tf = stats["weighted_word_freq"][word] / M_i
            # IDF计算（与之前相同）
            docs_with_word = sum(
                1 for c in class_stats
                if word in class_stats[c]["doc_contain_word"]
            )
            idf = math.log(total_train_docs / (docs_with_word + 1))
            prob = weighted_tf * idf
        else:
            # 平滑处理
            prob = 1 / (M_i + M_a)

        conditional_prob[cls][word] = prob


# 分类逻辑，模型测试
def classify_with_weight(text):
    """带权重的分类函数"""
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
                1 / (model["M_a"] + model["conditional_prob"][cls]["total_weighted_words"])
            )
            # 应用权重：P(W_k|C_i) × L_j
            weighted_prob = base_prob * weight
            log_prob += math.log(weighted_prob)

        if log_prob > max_log_prob:
            max_log_prob = log_prob
            best_class = cls

    return best_class

# ----------------------------
# 模型保存与测试
# ----------------------------
model = {
    "prior_prob": prior_prob,
    "conditional_prob": dict(conditional_prob),
    "M_a": M_a,
    "classes": list(class_stats.keys())
}

# 测试集评估
correct = 0
total = len(test_data)
for true_cls, html in test_data:
    pred_cls = classify_with_weight(html)
    if pred_cls == true_cls:
        correct += 1

accuracy = correct / total
print(f"改进后测试集准确率: {accuracy:.4f}")
