import requests
from bs4 import BeautifulSoup
import time
from loguru import logger
import re
import time
import random
import os
from tqdm import tqdm

receipe_type_menu_url = "https://home.meishichina.com/recipe-type.html"
headers = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0"
}

# 读取菜谱类型url
receipe_type_url_list = []
resp = requests.get(receipe_type_menu_url, headers=headers)
soup = BeautifulSoup(resp.text, "html.parser")
category_sub_list = soup.find_all(class_="category_sub")
## 遍历菜谱大类
for category_sub in category_sub_list:
    a_list = category_sub.find_all("a")
    for a in a_list:
        href = a["href"]
        receipe_type_url_list.append(href)
logger.info(f"菜谱大类有:{len(receipe_type_url_list)}种")
data_save_path = "/root/data/MeiShiTianXia"
os.makedirs(data_save_path, exist_ok=True)
receipe_url_list = []
# 遍历每一个菜谱类别网址
count = 0
for receipe_type_url in tqdm(receipe_type_url_list):
    count += 1
    if count <= 80:
        continue
    elif count > 100:
        break
    # if count > 20:
    #     break
    if "ribencai" in receipe_type_url:
        break
    scan_all = False
    page_id = 0
    logger.info(f"菜谱类别网址:{count}:{receipe_type_url}")
    while scan_all == False:
        resp = requests.get(receipe_type_url, headers=headers, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")
        ## 遍历当前页的数据
        receipe_wrapper = soup.find(class_="ui_newlist_1")
        receipe_detail_list = receipe_wrapper.find_all(class_="detail")
        for detail in receipe_detail_list:
            receipe_url = detail.find("h2").find("a")["href"]
            receipe_url_list.append(receipe_url)
        ## 下一页的url
        next_page_wrapper = soup.find(class_="ui-page-inner")
        next_page_a = next_page_wrapper.find_all("a")[-1]
        if next_page_a.has_attr('href') and next_page_a['href'] and page_id <= 30:
            receipe_type_url = next_page_a["href"]
            print(receipe_type_url)
            page_id += 1
            logger.info(f"第{page_id}页处理结束")
            sleep_time = random.uniform(5, 10)
            time.sleep(sleep_time)
        else:
            scan_all = True
    logger.info(f"共{page_id}页数据")

# 对食谱网址进行去重并保存
receipe_url_list = list(set(receipe_url_list))
logger.info(f"共有{len(receipe_url_list)}条食谱网址")
with open(os.path.join(data_save_path, "receipe_url.txt"), "a+") as f:
    for url in receipe_url_list:
        f.write(url + "\n")

# 遍历食谱界面
# for receipe_url in receipe_url_list:
#     resp2 = requests.get(receipe_url, headers=headers)
#     soup2 = BeautifulSoup(resp2.text, "html.parser")
#     ## 获得食品图片
#     img_url = soup2.find(class_="J_photo").find("img")["src"]
#     ## 获得标题
#     receipe_title = soup2.find(id="recipe_title")["title"]
#     ## 获得食材
#     ingredients_info = ""
#     recipDetail_list = soup2.find_all(class_="particulars")
#     for recipDetail in recipDetail_list:
#         title = recipDetail.find("legend").text
        
#         if title != "调料":
#             ingredients_info += f"{title}: "
#             category_list = recipDetail.find_all(class_="category_s1")
#             for category in category_list:
#                 ingred = category.text.strip()
#                 ingredients_info += f"{ingred},"
#             ingredients_info += "\n"
#     ## 获得步骤
#     recipeStep_wrapper = soup2.find(class_="recipeStep")
#     step_list = recipeStep_wrapper.find_all("li")
#     receipe_steps_info = ""
#     step_id = 0
#     for step_li in step_list:
#         step_id += 1
#         step_text = step_li.text.strip()
#         # 使用正则表达式去掉开头的数字序列
#         text_cleaned = re.sub(r'^\d+', '', step_text)
#         descrip = f"第{step_id}步: {text_cleaned}\n"
#         receipe_steps_info += descrip
#     logger.info(f"食物食谱网址:{receipe_url}")
#     logger.info(f"食物名称:{receipe_title}")
#     logger.info(f"食材:{ingredients_info}")
#     logger.info(f"步骤:{receipe_steps_info}")
#     break