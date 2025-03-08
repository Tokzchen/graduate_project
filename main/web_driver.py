import time

from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class WebAutomation:
    def __init__(self, browser='chrome', driver_path=None):
        if browser.lower() == 'chrome':
            service = Service(driver_path)
            self.driver = webdriver.Chrome(service=service)
            # 设置页面加载超时时间为 10 秒
            self.driver.set_page_load_timeout(10)
        elif browser.lower() == 'firefox':
            self.driver = webdriver.Firefox()
        else:
            raise ValueError("Unsupported browser type")

    def open_url(self, url):
        self.driver.get(url)

    def click(self, xpath, timeout=5):
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            element.click()
        except Exception as e:
            raise RuntimeError(f'点击元素{xpath}发生错误:{e}')

    def type(self, xpath, text_content, timeout=5):
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            element.clear()
            element.send_keys(text_content)
        except Exception as e:
            raise RuntimeError(f'输入到{xpath}时发生错误:{e}')

    def web_backward(self):
        self.driver.back()

    def get_web_content(self,attribute='html'):
        """
        获取网页（非网页元素）的内容
        :param attribute: title, current_url, html, name
        :return:
        """
        if attribute=='html':
            return self.driver.page_source
        elif attribute=='title':
            return self.driver.title
        elif attribute=='name':
            return self.driver.name
        elif attribute=='current_url':
            return self.driver.current_url

    def get_attribute(self, xpath, attribute_name):
        element = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return element.get_attribute(attribute_name)

    def maximize_window(self):
        self.driver.maximize_window()

    def get_window_handles(self):
        """获取所有窗口的句柄"""
        return self.driver.window_handles

    def switch_to_window(self, handle):
        """切换到指定的窗口"""
        self.driver.switch_to.window(handle)

    def get_current_window_handle(self):
        """获取当前窗口的句柄"""
        return self.driver.current_window_handle

    def close_window(self):
        """关闭当前窗口"""
        self.driver.close()

    def get_element_num(self, xpath):
        """获取匹配指定 XPath 的元素数量"""
        try:
            elements = self.driver.find_elements(By.XPATH, xpath)
            return len(elements)
        except Exception as e:
            raise RuntimeError(f'获取元素{xpath}数量时发生错误:{e}')

    def web_move_to(self, xpath):
        """将鼠标移动到指定元素上"""
        try:
            element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
            actions = ActionChains(self.driver)
            actions.move_to_element(element).perform()
        except Exception as e:
            raise RuntimeError(f'移动到{xpath}时发生错误:{e}')

    def web_refresh(self):
        """
        刷新当前网页
        :return:
        """
        self.driver.refresh()

    def close(self):
        """关闭浏览器"""
        self.driver.quit()

    def web_execute_js(self,js):
        """
        执行js代码
        :param js:
        :return:
        """
        try:
            self.driver.execute_script(js)
        except Exception as e:
            raise RuntimeError(f'执行js代码{js}报错:{e}')

    def web_click_js(self,xpath):
        """
        使用js的方式进行点击
        :param xpath:
        :return:
        """
        js=f"""document.evaluate('{xpath}',document).iterateNext().click()"""
        self.web_execute_js(js)

# 示例用法
if __name__ == "__main__":
    # 初始化 WebAutomation 对象，指定浏览器和 WebDriver 路径
    web = WebAutomation(browser='chrome', driver_path='../diver/132.0.6834.160/chromedriver.exe')

    # 打开网页
    web.open_url('https://news.qq.com/')
    web.maximize_window()
    # 使用 XPath 点击元素
    search_box='//div[contains(@class,"search-box")]/input'
    web.type(search_box,'暴雨灾害')
    time.sleep(5)

    # 关闭浏览器
    web.close()
