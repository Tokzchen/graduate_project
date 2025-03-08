# 测试用例
import re

import jieba
from bs4 import BeautifulSoup

from main.bayes_topic_scrawling import BayesTopicScrawling

test_html = """
<html>
    <body>
        <h1>暴&amp;雨灾&#害;预警</h1>
        <script>const data = 'AI大模型';</script>
        <p>人工智能大模型应对天气变化</p>
        <img alt="AI大模型分析暴雨灾害">
        <div>暴雨灾害越来越严重</div>
</body>
</html>
"""

keywords = ["暴雨灾害", "建筑物",'越来越严']
for kw in keywords:
    jieba.add_word(kw, freq=100000)
def check_html_exist_kw(html, keywords):
    # 使用BeautifulSoup提取可见文本（自动处理HTML实体）
    soup = BeautifulSoup(html, 'html.parser')

    # 移除脚本和样式内容
    for tag in soup(['script', 'style', 'noscript', 'meta', 'link']):
        tag.decompose()

    # 获取纯文本（保留alt/text等属性）
    text = soup.get_text(separator=' ', strip=True)

    # 执行精确分词
    words = jieba.lcut(text)
    print("【精确分词结果】", "/".join(words))

    # 生成检测结果
    return {kw: kw in words for kw in keywords}
print("📄 原始HTML片段：")
print(test_html.strip().replace('\n', ' '))
print("\n🔍 检测结果：")
result = check_html_exist_kw(test_html, keywords)
print({k: "✅存在" if v else "❌不存在" for k, v in result.items()})