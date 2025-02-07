#encoding=utf-8
# 获取网页特征向量
import copy
import pickle
import re
from bs4 import BeautifulSoup
import jieba
from collections import Counter, defaultdict

from src.top_relevance import TopicRelevance


class HtmlFeatureMatrix:
    def __init__(self,feature_setting=(2.0,1.8,1.8,1.0,1.5,2.5)):
        self.feature_setting=feature_setting
        with open('../src/graph_nodes.pkl', 'rb') as f:
            nodes = pickle.load(f)
            topic_relevance = TopicRelevance(nodes, nodes[1])
            topic_relevance.main_generate(nodes[1], nodes, (0.2, 0.2, 0.2, 0.2, 0.2))
            kw_list=topic_relevance.keyword_list
            for w in kw_list:
                jieba.add_word(w)

    def _extract_section(self, soup, tags_to_remove=[]):
        """提取并净化body内容"""
        if not soup.body:
            return ""

        body_copy = copy.copy(soup.body)
        for tag in body_copy.find_all(tags_to_remove):
            tag.decompose()
        return body_copy.get_text(separator=" ", strip=True)

    def analyze_html(self, html_text, keywords):
        """精确分类型统计词频"""
        soup = BeautifulSoup(html_text, 'html.parser')

        # 标题提取
        title = soup.title.get_text(strip=True) if soup.title else ""

        # 关键词提取
        meta_keywords = soup.find("meta", {"name": re.compile(r"^keywords$", re.I)})
        keywords_content = meta_keywords["content"] if meta_keywords else ""

        # 描述提取（兼容OG协议）
        descriptions = []
        for meta in soup.find_all("meta"):
            if meta.get("name", "").lower() == "description" or \
                    meta.get("property", "").lower() == "og:description":
                descriptions.append(meta.get("content", ""))
        description_content = " ".join(filter(None, descriptions))

        # 标题层级提取（包含header容器）
        hl_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header'])
        hl_content = " ".join([t.get_text(strip=True) for t in hl_tags])

        # 锚文本提取（含规范化链接）
        anchors = []
        for a in soup.find_all('a'):
            anchors.append(a.get_text(strip=True))  # 锚文本内容
        for link in soup.find_all('link', {'rel': 'canonical'}):
            if link.get('href'):
                anchors.append(link['href'])  # 规范化链接
        anchor_content = " ".join(anchors)

        # 正文提取（排除已明确分类的内容）
        body_content = self._extract_section(soup,
                                             tags_to_remove=['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header', 'a', 'link'])

        categories = {
            "title": title,
            "keywords": keywords_content,
            "description": description_content,
            "hl": hl_content,
            "anchor": anchor_content,
            "body": body_content,
        }

        # 词频统计逻辑
        results = {k: defaultdict(int) for k in categories}
        max_freq = defaultdict(int)

        for category, text in categories.items():
            words = jieba.lcut(text)
            counter = Counter(words)

            for kw in keywords:
                count = counter.get(kw, 0)
                results[category][kw] = count
                if count > max_freq[kw]:
                    max_freq[kw] = count

        return results, max_freq

    def main_generate(self, html_text, keywords):
        """生成加权特征向量"""
        results, max_freq = self.analyze_html(html_text, keywords)
        feature_vector = []

        for kw in keywords:
            total = 0.0
            max_count = max_freq.get(kw, 0)

            # 对各分类进行加权计算
            for i, category in enumerate([
                "title", "keywords", "description",
                "hl", "anchor", "body"
            ]):
                count = results[category].get(kw, 0)
                if max_count == 0:  # 防零除
                    ratio = 0.0
                else:
                    ratio = (count + 1) / (max_count + 1)
                total += ratio * self.feature_setting[i]

            feature_vector.append(round(total, 4))

        return feature_vector


if __name__ == '__main__':
    t="""<html xmlns="http://www.w3.org/1999/xhtml"><head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8">
<title>暴雨_国家应急广播网</title>
<meta name="keywords" content="国家应急广播 国家应急广播">
<meta name="description" content="国家应急广播 国家应急广播">
<meta name="filetype" content="0">
<meta name="publishedtype" content="1">
<meta name="pagetype" content="2">
<meta name="catalogs" content="PAGE1513775405516841">  
<script type="text/javascript" src="http://www.cneb.gov.cn/contentpage2013/js/jquery-1.7.2.min.js"></script><link type="text/css" rel="stylesheet" href="http://www.cneb.gov.cn/cneb/kpkxq/by/style/style.css?7c52f6e797c6e30729184b79b88f760d">
<script src="http://img.cneb.gov.cn/photoAlbum/templet/common/DEPA1480556202442298/main_20161201.js"></script>
<script type="text/javascript" src="http://www.cneb.gov.cn/contentpage2013/js/ckplayer/ckplayer1.js"></script>
<script type="text/javascript" src="http://www.cneb.gov.cn/contentpage2013/js/audioplayer/soundmanager2-nodebug-jsmin.js"></script>
<style type="text/css">
video::-internal-media-controls-download-button {
	display:none !important;
}
video::-webkit-media-controls-toggle-closed-captions-button {
	display:none !important;
}
video::-internal-media-controls-overflow-button {
	display:none !important;
}
video::-webkit-media-controls-enclosure {
    overflow:hidden;
	width:calc(100% + 32px);
	margin-left:auto;
}
video::-webkit-media-controls {
	overflow:hidden !important;
}	
</style><script type="text/javascript" referrerpolicy="no-referrer-when-downgrade" src="//d.webterren.com/common.js?z=27&amp;t=202304110258"></script></head>

<body style="min-width: 1000px;">
<!--#include virtual="/cneb/include_all_2014/gaiban/topbar/index.shtml"-->
<div id="page_head">
 <!--#include virtual="/cneb/include_all_2014/include_all_2014/nav/index.shtml"-->
</div>
<div id="page_body">

	<div class="wrapper_1000" id="SUBD1513775641215987">
		<div class="crumbs">
		<span><a href="http://www.cneb.gov.cn/" target="_blank">国家应急广播</a></span>&gt;
		
		
		<span class="last"><a href="http://www.cneb.gov.cn/kepuku/" target="_blank">科普库</a></span> &gt; <span class="last">暴雨</span>
		
		
		
</div>
	</div>

<div class="body_banner">
	<div class="banner">
	
		<span class="logo"><img src="http://img.cneb.gov.cn/photoAlbum/page/performance/img/2017/12/21/1513796235406_12.png" alt=""></span>
		<span class="title"><img src="http://img.cneb.gov.cn/photoAlbum/page/performance/img/2017/12/21/1513796226905_304.png" alt=""></span>
		<span class="brief"><img src="http://img.cneb.gov.cn/photoAlbum/page/performance/img/2019/5/30/1559184390504_660.png" alt=""></span>
	
	</div>
</div>
<style type="text/css">
.body_banner{height:143px;background:url(http://img.cneb.gov.cn/photoAlbum/templet/special/PAGE1513775405516841/ELMT1513775641215991_1516939633.jpg) no-repeat center top;}
</style>
	<div class="body_content" id="SUBD1513775641215996">
		
	<div class="wrapper_1000" id="SUBD1513775641215999">
		<div class="nav-item">
	
		<span class="nav_tips first cur"><i class="tips">一、读懂预警 防患未然</i></span>
	
		<span class="nav_tips "><i class="tips">二、主要危害及应对</i></span>
	
		<span class="nav_tips "><i class="tips">三、灾后注意事项</i></span>
	
</div>
<script type="text/javascript">

	var navDemo = $('.nav-item');
	var topHeight = 0;

	navDemo.find('.nav_tips').click(function () {
		var index = $(this).index();
		$(this).addClass("cur").siblings().removeClass("cur");
		var offsetHeight = $('.conbox_demo .conbox_tips').eq(index).offset().top;
		$("html:not(:animated),body:not(:animated)").animate({scrollTop: offsetHeight},300);
	});
</script><div class="ELMT1513775641215103">
<div class="vspace"></div>
</div>
<div class="conbox-item">
	
		<div class="col_w724" id="SUBD1513775641215107">
			<div class="conbox_demo">
    
	<div class="conbox_tips">
		<div class="top">
			<i class="logo">一</i>
			<div class="top_hd">
				<span class="title cur"><a href="javascript:;">读懂预警 防患未然</a></span>
			</div>
		</div>
	    <!--#include virtual="/cneb/kpkxq/by/11/index.shtml"-->
<!--#include virtual="/cneb/kpkxq/by/12/index.shtml"-->
<!--#include virtual="/cneb/kpkxq/by/13/index.shtml"-->
	</div>
	
	<div class="conbox_tips">
		<div class="top">
			<i class="logo">二</i>
			<div class="top_hd">
				<span class="title cur"><a href="javascript:;">主要危害及应对</a></span>
			</div>
		</div>
	    <!--#include virtual="/cneb/kpkxq/by/211/index.shtml"-->
<!--#include virtual="/cneb/kpkxq/by/222/index.shtml"-->
<!--#include virtual="/cneb/kpkxq/by/233/index.shtml"-->
	</div>
	
	<div class="conbox_tips">
		<div class="top">
			<i class="logo">三</i>
			<div class="top_hd">
				<span class="title cur"><a href="javascript:;">灾后注意事项</a></span>
			</div>
		</div>
	    <!--#include virtual="/cneb/kpkxq/by/331/index.shtml"-->
<!--#include virtual="/cneb/kpkxq/by/332/index.shtml"-->
<!--#include virtual="/cneb/kpkxq/by/333/index.shtml"-->
	</div>
	
</div>
<!--<script type="text/javascript">
    document.oncontextmenu=function(){return false;}
</script>
-->
		</div>
	
	
		<div class="col_w259" id="SUBD1513775641215111">
			<div class="box_video">
	
		<div class="box_head">
			<span>暴雨</span>
		</div>
		<div class="box_body">
			<h3></h3>
			<p class="brief"></p>
			<div class="video" id="a3" datamp4="http://video.cneb.gov.cn/videostorage/convert/2017/07/21/78461500617962188.mp4" datajpg="http://img.cneb.gov.cn/photoAlbum/page/performance/img/2018/7/25/1532500895166_91.jpg" style="background-color: rgb(0, 0, 0); width: 240px; height: 140px; cursor: pointer;"><video controls="controls" src="http://video.cneb.gov.cn/videostorage/convert/2017/07/21/78461500617962188.mp4" id="ckplayer_a3" width="240px" height="140px" preload="metadata" poster="http://img.cneb.gov.cn/photoAlbum/page/performance/img/2018/7/25/1532500895166_91.jpg"></video></div>
			<div class="textbox">　　暴雨是我国主要气象灾害之一，长时间的暴雨容易导致山洪爆发、水库垮坝、江河横溢，给国民经济和人民生命财产带来严重危害。<br>
　　受季风气候影响，我国除西北个别省区外，其他地区都有出现暴雨的可能。4～6月，华南地区暴雨频发；6～7月，长江中下游常有持续性暴雨，历时长、面积广、雨量大，易导致洪涝灾害发生；7～8月，北方各省易发暴雨；8～10月，雨带又逐渐南撤。</div>
			<span id="details_video" class="details">查看详情</span>	
		</div>
	
</div>
<script type="text/javascript">
	$(function () {
		$('#details_video').click(function () {
			$(this).css('display','none').siblings('.textbox').css({'height': 'auto'});
		});
	});

   var mp4 = $('.box_video .box_body .video').attr('datamp4');
   var jpg = $('.box_video .box_body .video').attr('datajpg');
   var flashvars={
    f:mp4,
    c:0,
    p:2,
    b:0,
    i:jpg,
    my_url:encodeURIComponent(window.location.href)
  };
   var params={bgcolor:'#FFF',allowFullScreen:true,allowScriptAccess:'always',wmode:'transparent'};
   var vide=[mp4+'->video/mp4'];   
  CKobject.embed('http://www.cneb.gov.cn/contentpage2013/js/ckplayer/ckplayer.swf ','a3','ckplayer_a3','240px','140px',true,flashvars,vide,params);
</script>     <div class="vpsace" style="height:20px;"></div>
<div class="mu_boxp">
	<div class="box_list" id="box_list">
		<div class="mu_box">
			<div class="con">
				<ul>
				
					<li>读懂预警 防患未然</li>
				
					<li>主要危害及应对</li>
				
					<li>灾后注意事项</li>
				
				</ul>
			</div>
		</div>
		<div class="vspace" style="height:20px;"></div>
		<div class="back_top" id="back_top"><a href="javascript:void(0)">返回顶部</a></div>
	</div>
 </div>
<div class="vspace"></div>
<script type="text/javascript">
	$(function() {
		var demolis=$(".conbox_demo .conbox_tips");
		var mimuli=$(".box_list .mu_box .con ul li")
		var demoarr=[];
		jsplay();

		function sidebar(){
			var t = document.documentElement.scrollTop || document.body.scrollTop;
			var clientHeight = document.documentElement.clientHeight || document.body.clientHeight
			var w = $(window).width();
			var w_right = Math.ceil((w - 1000)/2 - 48);
			var box_listt=$(".mu_boxp").offset().top
			var arrynus=demoarr.length;
			for(var i= 0;i<arrynus;i++){
				if(t>=demoarr[i] && t<demoarr[i+1]){
					mimuli.eq(i).addClass("attr").siblings().removeClass("attr");
				}
			}



			if(t>box_listt){
				$("#box_list").attr({"class":"box_list attr"});
			}
			else{
				$("#box_list").attr({"class":"box_list"});
			}



		}
		sidebar();
		var throldHold = 100;
		window.onresize = window.onscroll = function () {
			clearTimeout(throldHold);
			window.timer = setTimeout(sidebar, throldHold);
		}

		$("#back_top").click(function(){
			$('body,html').animate({scrollTop:0},300)
		})


		function jsplay(){
			demoarr=[];
			demolis.each(function(i){
				var sj=demolis.eq(i).offset().top
				demoarr.push(sj)
			})
		}

		mimuli.click(function(){



			var rwlis=$(this).index();
			var sctt=demoarr[rwlis];
			$('body,html').animate({scrollTop:sctt},300,function(){
				mimuli.eq(rwlis).addClass("attr").siblings().removeClass("attr");
			})

		})

		$('#details').click(function () {
			$(this).css('display','none').siblings('.textbox').css({'height': 'auto'});
		});
		var contDemo = $('.conbox_demo');

		contDemo.find('.conbox_tips').each(function () {
			$(this).find('.top .title').click(function () {
				var flag = $(this).hasClass('cur');
				if (flag) {
					$(this).removeClass('cur');
					$(this).parents('.conbox_tips').find('.top').siblings('.tboxs').css('display','none');
					$(this).parents('.conbox_tips').find('.vspace').css('display','none');
				}else {
					$(this).addClass('cur');
					$(this).parents('.conbox_tips').find('.top').siblings('.tboxs').css('display','block');
					$(this).parents('.conbox_tips').find('.vspace').css('display','block');
				}
				jsplay();

			})
		});

		contDemo.find('.tboxs').each(function () {
			$(this).find('.tboxs_title .tl_info').click(function () {
				var sign = $(this).parents('.tboxs_title').hasClass('cur');
				if (sign) {
					$(this).parents('.tboxs_title').removeClass('cur');
				}else {
					$(this).parents('.tboxs_title').addClass('cur');
				}
				jsplay();
			});
		});

		/* nav and content code */
		var navDemo = $('.nav-item');
		var topHeight = 0;

		navDemo.find('.nav_tips').click(function () {
			var index = $(this).index();
			$(this).addClass("cur").siblings().removeClass("cur");
			var offsetHeight = $('.conbox_demo .conbox_tips').eq(index).offset().top;
			$("html:not(:animated),body:not(:animated)").animate({scrollTop: offsetHeight},300);
		});




	});
</script>
		</div>
	
	<div class="clear"></div>
</div>
	</div>


	</div>

</div>
<!--#include virtual="/cneb/include_all_2014/gaiban/shouyeyejiao/index.shtml"-->
<!-- webTerren -->
<div style="display:none">
<script type="text/javascript">document.write(unescape("%3Cscript src='http://cl2.webterren.com/webdig.js?z=27' type='text/javascript'%3E%3C/script%3E"));</script><script src="http://cl2.webterren.com/webdig.js?z=27" type="text/javascript"></script>
<script type="text/javascript">wd_paramtracker("_wdxid=000000000000000000000000000000000000000000")</script>
</div>
<!-- webTerren -->
<script type="text/javascript">var cnzz_protocol = (("https:" == document.location.protocol) ? " https://" : " http://");document.write(unescape("%3Cspan id='cnzz_stat_icon_1260222408'%3E%3C/span%3E%3Cscript src='" + cnzz_protocol + "s95.cnzz.com/z_stat.php%3Fid%3D1260222408%26show%3Dpic1' type='text/javascript'%3E%3C/script%3E"));</script><span id="cnzz_stat_icon_1260222408"></span><script src=" http://s95.cnzz.com/z_stat.php?id=1260222408&amp;show=pic1" type="text/javascript"></script>

 </body></html>"""
    with open('graph_nodes.pkl','rb')as f:
        nodes=pickle.load(f)
        topic_relevance=TopicRelevance(nodes,nodes[1])
        topic_relevance.main_generate(nodes[1],nodes,(0.2,0.2,0.2,0.2,0.2))
        print(topic_relevance.topic_meaning_matrix)
        print(topic_relevance.keyword_list)
        v=HtmlFeatureMatrix().main_generate(t,topic_relevance.keyword_list)
        print(v)


