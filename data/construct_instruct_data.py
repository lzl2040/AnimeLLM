import json
import random

json_path = "/root/data/MeiShiTianXia/dataset.json"
# 读取包含食品信息的 JSON 文件
with open(json_path, 'r', encoding='utf-8') as file:
    food_data = json.load(file)

# 定义问题列表
questions_about_name = [
    "这道菜的名称是什么？",
    "图片中这道菜的名称是什么？",
    "你能告诉我这道菜的名字吗？",
    "这是什么菜？",
    "你能告诉我这道菜的名称吗？",
    "这道菜的烹饪名称是什么？",
    "你能提供这道菜的名称吗？",
    "图片中展示的这道菜属于哪个类别？",
    "你能识别出照片中展示的这道菜吗？",
    "图片中描绘的是哪道菜？"
]

questions_about_ingredients = [
    "你能识别出这张图片中的食材吗？",
    "这张图片中有哪些食材？",
    "请检测这张照片中的食材。",
    "你能从这张照片中分辨出哪些食材吗？",
    "你能从这张图片中识别出食材吗？",
    "图片中这道菜的食材是什么？",
    "你能列出照片中这道菜的成分吗？",
    "图片中的这道菜是由什么组成的？"
]

questions_about_steps = [
    "你能详细说明这道菜的制作步骤吗？",
    "你能提供一个重现这道菜的食谱或指南吗？",
    "你能提供一个重现这道菜的指南吗？",
    "你能提供一个重现这道菜的食谱吗？",
    "你能详细说明这道菜的烹饪步骤吗？",
    "怎么制作这道菜？",
    "如何制作这道菜？",
    "你能提供烹饪这道菜的说明吗？",
    "制作图片中这道菜的步骤或说明是什么？",
    "如何准备照片中展示的这道菜？",
    "你能描述图片中这道菜的食谱或制作过程吗？"
]

# 构建指令数据集
instruction_dataset = []

for food in food_data:
    name = food['name']
    ingredients = food['ingredients']
    steps = food['steps']
    img_path = f"images/{food['id']}/food.jpg"
    

    conversation = {
        "image": img_path,
        "conversations": [
            {
                "from": "human",
                "value": f"<image>\n{random.choice(questions_about_name)}"
            },
            {
                "from": "gpt",
                "value": f"这道菜是：{name}。"
            },
            {
                "from": "human",
                "value": random.choice(questions_about_ingredients)
            },
            {
                "from": "gpt",
                "value": f"这道菜包含的食材有：{ingredients}。"
            },
            {
                "from": "human",
                "value": random.choice(questions_about_steps)
            },
            {
                "from": "gpt",
                "value": f"制作这道菜的步骤是：{steps}。"
            }
        ]
    }

    instruction_dataset.append(conversation)

# 将指令数据集保存为 JSON 文件
with open("/root/data/MeiShiTianXia/instruction_dataset.json", 'w', encoding='utf-8') as outfile:
    json.dump(instruction_dataset, outfile, ensure_ascii=False, indent=4)

print("指令数据集构建完成并已保存为 instruction_dataset.json")