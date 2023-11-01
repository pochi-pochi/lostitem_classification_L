import requests
from time import sleep
from selenium import webdriver
# from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import os

# google画像検索にアクセス
service = Service(executable_path='C:/chromedriver-win64/chromedriver.exe')
options = webdriver.ChromeOptions()
options.add_argument("--headless")
browser = webdriver.Chrome(service=service, options=options)
url = "https://www.google.co.jp/imghp?hl=ja"
browser.get(url)

# 保存先のディレクトリ
base_directory = "dataset/train/Bags"

# 検索キーワードとしてフォルダ名を使用
actor_name = "Bags"

selector = "body > div.L3eUgb > div.o3j99.ikrT4e.om7nvf > form > div:nth-child(1) > div.A8SBwf > div.RNNXgb > div > div.a4bIc > textarea"
kw_search = browser.find_element(By.CSS_SELECTOR, selector)
kw_search.clear()
kw_search.send_keys(str(actor_name))
kw_search.send_keys(Keys.ENTER)

# BeautifulSoupで画像検索したページの画像を取得する
cur_url = browser.current_url
res = requests.get(cur_url)
soup = BeautifulSoup(res.text, "html5lib")

img_tags = soup.find_all("img", limit=100)
img_urls = []

for img_tag in img_tags:
    url_a = img_tag.get("src")
    if url_a != None:
        img_urls.append(url_a)

# 画像を指定のフォルダに保存
save_dir = os.path.join(base_directory, actor_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
a = 1
for elem_url in img_urls:
    try:
        r = requests.get(elem_url)
        with open(os.path.join(save_dir,
                                f"{actor_name}画像{a}.jpg"), "wb") as fp:
            fp.write(r.content)
        a += 1
        sleep(0.1)
    except:
        pass

browser.quit()
