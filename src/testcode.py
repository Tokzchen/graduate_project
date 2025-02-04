import re

import jieba

stopwords = set(["的", "了", "在", "是", "和", ...])
def preprocess_html(html_text):
    # 简单去除HTML标签（实际可能需要更复杂的清洗）
    text = re.sub(r'<[^>]+>', '', html_text)
    # 使用jieba分词并过滤停用词
    words = [word for word in jieba.lcut(text) if word not in stopwords]
    return words

s="""<html>
 <head>
  <title>
   2平 nba季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂
  </title>
  <meta content=""2平, nba季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂, 保罗, 兰德里, 加索尔"" name=""keywords""/>
  <meta content=""黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂，此前的比赛中，双方战成2-2平，因此本场比赛对于两支球队来说都非常重要，"" name=""description""/>
 </head>
 <body>
  <h1>
   2平
  </h1>
  <p>
   黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战 新浪体育讯北京时间4月27日，NBA季后赛首轮洛杉矶湖人主场迎战新奥尔良黄蜂，此前的比赛中，双方战成2-2平，因此本场比赛对于两支球队来说都非常重要，赛前双方也公布了首发阵容：湖人队：费舍尔、科比、阿泰斯特、加索尔、拜纳姆黄蜂队：保罗、贝里内利、阿里扎、兰德里、奥卡福[新浪NBA官方微博][新浪NBA湖人新闻动态微博][新浪NBA专题][黄蜂vs湖人图文直播室](新浪体育)
  </p>
 </body>
</html>"""

print(preprocess_html(s))