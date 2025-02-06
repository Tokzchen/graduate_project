import collections
from urllib.parse import urlparse

s='https://www.51miz.com/so-sucai/4304092.html?utm_term=24813233&utm_source=baidu&bd_vid=6067477385671942740'
s1='http://www.byzh.org.cn/article/doi/10.12406/byzh.2023-239?viewType=HTML'

def normalize_url(url):
    """
    标准化 URL，移除尾斜杠和查询参数等。
    :param url: 原始 URL (str)
    :return: 标准化后的 URL (str)
    """
    parsed = urlparse(url)
    # 移除查询参数和 fragment（锚点）
    normalized_url = parsed.scheme + "://" + parsed.netloc + parsed.path
    # 去掉尾部的斜杠
    if normalized_url.endswith('/'):
        normalized_url = normalized_url[:-1]
    return normalized_url

st=set()
st.add(1)
st.add(2)
st.add(2)
print(len(st))