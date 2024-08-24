import requests
import os
from loguru import logger
from bs4 import BeautifulSoup
import time
import re
from tqdm import tqdm
import urllib
import json
import random

def save_to_txt(save_path, content):
    with open(save_path, "w") as f:
        f.write(content)

headers = {
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0"
}
data_save_path = "/root/data/MeiShiTianXia"
img_save_dir = os.path.join(data_save_path, "images")
os.makedirs(img_save_dir, exist_ok=True)
receipe_url_list = []
with open(os.path.join(data_save_path, "receipe_url.txt"), "r") as f:
    for line in f:
        receipe_url_list.append(line.strip())
# 去重
receipe_url_list = sorted(list(set(receipe_url_list)))
logger.info(f"共有{len(receipe_url_list)}条食谱网址")
# 遍历食谱界面
count = 0
data_list = []
for receipe_url in tqdm(receipe_url_list):
    count += 1
    if count <= 18800:
        continue
    if count % 200 == 0:
        time.sleep(300)
        break
    ## 获得食谱的唯一标识符
    r_id = int(receipe_url.split("-")[-1].split(".")[0])
    print(r_id)
    # if r_id != 611823:
    #     continue
    # else:
    #     print(f"{count}")
    #     break
    html = requests.get(receipe_url, headers=headers).text
    ## 防止乱码
    html = html.encode("ISO-8859-1")
    html = html.decode("utf-8")
    # print(html)
    soup2 = BeautifulSoup(html, "html.parser")
    ## 获得食品图片
    img_url = soup2.find(class_="J_photo").find("img")["src"]
    # print(img_url)
    ## 获得标题
    receipe_title = soup2.find(id="recipe_title")["title"]
    ## 获得食材
    ingredients_info = ""
    recipDetail_list = soup2.find_all(class_="particulars")
    for recipDetail in recipDetail_list:
        title = recipDetail.find("legend").text
        
        if title != "调料":
            ingredients_info += f"{title}: "
            category_list = recipDetail.find_all(class_="category_s1")
            for category in category_list:
                ingred = category.text.strip()
                ingredients_info += f"{ingred},"
            ingredients_info += "\n"
    ## 获得步骤
    recipeStep_wrapper = soup2.find(class_="recipeStep")
    step_list = recipeStep_wrapper.find_all("li")
    receipe_steps_info = ""
    step_id = 0
    for step_li in step_list:
        step_id += 1
        step_text = step_li.text.strip()
        # 使用正则表达式去掉开头的数字序列
        text_cleaned = re.sub(r'^\d+', '', step_text)
        descrip = f"第{step_id}步: {text_cleaned}\n"
        receipe_steps_info += descrip
    # logger.info(f"食物食谱网址:{receipe_url}")
    logger.info(f"食物名称:{receipe_title}:{receipe_url}")
    # logger.info(f"食材:{ingredients_info}")
    # logger.info(f"步骤:{receipe_steps_info}")
    # 图片保存地址
    try:
        img_resp = requests.get(img_url, headers=headers)
        # 检查响应状态码
        img_resp.raise_for_status()  # 如果响应状态码不是 200，会抛出 HTTPError
        save_path = os.path.join(img_save_dir, f"{r_id}")
        os.makedirs(save_path, exist_ok=True)
        img_save_path = os.path.join(save_path, "food.jpg")
        with open(img_save_path,'wb') as file_obj:
            file_obj.write(img_resp.content)
        # 保存食物名称
        name_save_path = os.path.join(save_path, "name.txt")
        save_to_txt(name_save_path, receipe_title)
        # 保存食材
        ingre_save_path = os.path.join(save_path, "ingredients.txt")
        save_to_txt(ingre_save_path, ingredients_info)
        # 保存步骤
        step_save_path = os.path.join(save_path, "steps.txt")
        save_to_txt(step_save_path, receipe_steps_info)
        data_dict = {}
        data_dict["img_path"] = img_save_path
        data_dict["name"] = receipe_title
        data_dict["ingredients"] = ingredients_info
        data_dict["steps"] = receipe_steps_info
        data_list.append(data_dict)
    except requests.exceptions.RequestException as e:
        # 处理请求异常（例如网络问题、无效 URL、HTTP 错误等）
        print(f"请求 {img_url} 时出错: {e}")
        img_resp = None  # 将 img_resp 设置为 None 或其他适当的默认值
    # img_resp = requests.get(img_url, headers=headers)
    
    sleep_time = random.uniform(5, 10)
    time.sleep(sleep_time)

# 保存到json文件
with open(os.path.join(data_save_path, "dataset.json"), 'w', encoding='utf-8') as json_file:
    json.dump(data_list, json_file, ensure_ascii=False, indent=4)