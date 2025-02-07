import httpx


def get_html_text_from_url( url, retry):
    """
    从url获取html文本
    :param url: 目标url
    :param retry: 重试次数
    :return:
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
    }

    with httpx.Client() as session:
        cnt = retry
        while cnt > 0:
            try:
                response = session.get(url, headers=headers, timeout=10, follow_redirects=True)
                if response.status_code == 200:
                    return response.text
                else:
                    print(f"请求失败，状态码: {response.status_code}")
            except httpx.RequestError as e:
                print(f"请求出错: {e}")
            cnt -= 1

    raise RuntimeError(f"请求 {url} 失败")

r=get_html_text_from_url('https://baike.baidu.com/item/2024%E5%B9%B4%E8%BE%BD%E5%AE%81%E6%9A%B4%E9%9B%A8/64727038?fr=aladdin',3)
print(r)