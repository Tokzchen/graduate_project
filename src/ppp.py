from src.link_rating import LinkRating
from src.top_relevance import TopicRelevance

if __name__ == "__main__":
    # 示例 HTML 文本
    html_text = """
    <html>
        <body>
            <a href="weather.html">最新天气状况</a>
            <a href="alerts.html">预警信息</a>
            <a href="temperature.html">气温预测</a>
            <a href="other.html">其他信息</a>
            <a href="alerts.html">预警预警预警</a>
        </body>
    </html>
    """


def test_pagerank():
    graph = LinkRating()

    # 模拟网页链接关系和锚文本相关度
    graph.add_page_to_graph("A", ["B", "C"], {"B": 0.8, "C": 0.6})
    graph.add_page_to_graph("B", ["C", "D"], {"C": 0.7, "D": 0.5})
    graph.add_page_to_graph("C", ["A"], {"A": 0.9})
    graph.add_page_to_graph("D", ["A", "B"], {"A": 0.4, "B": 0.3})

    # 初始化 PageRank 值
    for url in ["A", "B", "C", "D"]:
        graph.pagerank[url] = 1.0  # 初始值为 1

    # 计算 PageRank
    damping = 0.85
    omega = 0.5

    for url in ["A", "B", "C", "D"]:
        pr = graph.get_pagerank(url, damping, omega)
        print(f"PageRank of {url}: {pr:.4f}")

tp=TopicRelevance()
tm=tp.main_generate()
test_pagerank()