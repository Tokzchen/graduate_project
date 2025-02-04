# coding=utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup

from src.html_feature_matrix import HtmlFeatureMatrix


def extract_keywords(text, top_n=5):
    """
    使用TF-IDF提取文本中的关键词
    :param text: 输入文本
    :param top_n: 提取的关键词数量
    :return: 关键词列表
    """
    vectorizer = TfidfVectorizer(max_features=top_n)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    return feature_names

def text_to_html(text, keywords):
    """
    将文本转换为HTML，并根据关键词分配权重
    :param text: 输入文本
    :param keywords: 关键词列表
    :return: 生成的HTML文本
    """
    # 初始化BeautifulSoup对象
    soup = BeautifulSoup(features="html.parser")

    # 创建完整的HTML结构
    html = soup.new_tag("html")
    soup.append(html)

    # 创建<head>标签
    head = soup.new_tag("head")
    html.append(head)

    # 生成<title>标签（最高权重）
    title = soup.new_tag("title")
    title.string = " ".join(keywords[:2])  # 取前两个关键词作为标题
    head.append(title)

    # 生成<meta>标签（次高权重）
    meta_keywords = soup.new_tag("meta", attrs={"name": "keywords", "content": ", ".join(keywords)})
    head.append(meta_keywords)

    meta_description = soup.new_tag("meta", attrs={"name": "description", "content": text[:100]})  # 取前100字符作为描述
    head.append(meta_description)

    # 创建<body>标签
    body = soup.new_tag("body")
    html.append(body)

    # 生成<h1>标签（高权重）
    h1 = soup.new_tag("h1")
    h1.string = keywords[0]  # 取第一个关键词作为主标题
    body.append(h1)

    # 生成<p>标签（基础权重）
    p = soup.new_tag("p")
    p.string = text  # 正文内容
    body.append(p)

    return soup.prettify()

def main():
    # 示例文本
    text = """
    休斯顿公开赛尼蒂斯暂领先 米克尔森77杆面临淘汰
　　新浪体育讯　北京时间4月4日，世界第二米克尔森在休斯顿公开赛上遭遇滑铁卢，第一轮仅仅打出77杆，很有可能在红石高尔夫俱乐部遭遇淘汰。目前暂时领先的选手是詹姆士-尼蒂斯(James Nitties)。
　　休斯顿时间4月2日星期四遭遇强风，休斯顿公开赛第一轮不得不延迟到今天进行。米克尔森昨天便很挣扎，今天比赛恢复之后也没有好转。他最终使用了31推，打出77杆，高于标准杆5杆。休斯顿公开赛第一轮目前还没有结束。会馆领先者为詹姆士-尼蒂斯，他打出66杆，低于标准杆6杆。也就是说米克尔森18洞过后便落后了11杆。
　　77杆是米克尔森2009赛季的最高杆数，也是自2008年圆石滩职业/业余配对赛以来的最高杆数。去年他在圆石滩第三轮打出78杆之后遭遇淘汰。从现在的情况来看，米克尔森也很容易被淘汰。
　　米克尔森今年的整体表现不错，总共赢得了两场比赛，包括不久之前在迈阿密赢得的WGC-CA锦标赛。在赢得那一场世锦赛之后，米克尔森的世界排名积分非常接近老虎伍兹。没有想到老虎伍兹上一周在湾丘俱乐部赢得了复出之后的第一场比赛：帕尔默邀请赛，及时拉开了双方的差距。很显然，如果米克尔森在红石高尔夫俱乐部被淘汰，他们之间的比分差距将进一步加大。他要想追上老虎伍兹，第一次拿到世界第一将更难。
　　或许这还不是最糟糕的事情。米克尔森把休斯顿公开赛视为热身赛，积极备战下一周的美国名人赛。这一周表现不佳，肯定对他的信心是一个不小的打击。
　　詹姆士-尼蒂斯是美巡赛新人，许多人都对他不熟悉。可是相信不久之后，人们便会逐渐认识他，因为最近一段时间詹姆士-尼蒂斯的表现都很不错。仅仅第三次参加美巡赛，詹姆士-尼蒂斯便在FBR公开赛上获得了并列第四名，随后在墨西哥举行的悦榕庄精英赛上又拿到了并列第六名。最近的三场比赛，他有两场都获得了并列第22名，其中包括上一周的帕尔默邀请赛。在美巡赛新人排名中，詹姆士-尼蒂斯目前位于第四位。一个胜利不仅能把他送到榜首位置，也会将他送到奥古斯塔，因为美国名人赛为休斯顿公开赛冠军准备最后一份邀请函。
　　从后九洞开球，詹姆士-尼蒂斯第一轮的表现很出色，仅仅只错过了3个果岭，以小鸟-小鸟-帕结束本轮比赛。其中第七洞拿下33英尺小鸟推应该是他今天的精彩表现之一。
　　(小风)
    """

    # 提取关键词
    keywords = extract_keywords(text, top_n=5)
    print("提取的关键词:", keywords)

    # 转换为HTML
    html_output = text_to_html(text, keywords)
    return html_output

if __name__ == "__main__":
    html_output=main()
    print(html_output)
    keywords_list=['体育','淘汰','出色']
    final_matrix=HtmlFeatureMatrix().main_generate(html_output,keywords_list)
    print(final_matrix)