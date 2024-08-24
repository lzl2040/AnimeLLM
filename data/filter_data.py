import os
import json
from PIL import Image

def extract_info_from_folder(folder_path):
    data = []
    
    # 遍历images文件夹下的所有子文件夹
    for folder_name in os.listdir(folder_path):
        folder_full_path = os.path.join(folder_path, folder_name)
        
        if os.path.isdir(folder_full_path):
            # 构建字典来存储信息
            img_path = os.path.join(folder_full_path, "food.jpg")
            try:
                # 尝试打开图像文件
                with Image.open(img_path) as img:
                    # 在此处进行所需的图像处理，例如加载到内存或其他操作
                    entry = {
                        "id": folder_name,  # 子文件夹的名字作为id
                        "img_path": os.path.join(folder_full_path, "food.jpg"),
                        "name": None,
                        "ingredients": None,
                        "steps": None
                    }
            
                # 读取name.txt
                name_file_path = os.path.join(folder_full_path, "name.txt")
                # print(name_file_path)
                if os.path.exists(name_file_path):
                    with open(name_file_path, "r", encoding="utf-8") as name_file:
                        entry["name"] = name_file.read().strip()
                # 菜名长度超过12过滤
                if len(entry["name"]) > 12:
                    continue
                
                # 读取ingredients.txt
                ingredients_file_path = os.path.join(folder_full_path, "ingredients.txt")
                if os.path.exists(ingredients_file_path):
                    with open(ingredients_file_path, "r", encoding="utf-8") as ingredients_file:
                        entry["ingredients"] = ingredients_file.read().strip()
                
                # 读取steps.txt
                steps_file_path = os.path.join(folder_full_path, "steps.txt")
                if os.path.exists(steps_file_path):
                    with open(steps_file_path, "r", encoding="utf-8") as steps_file:
                        entry["steps"] = steps_file.read().strip()
                
                # 将字典添加到列表中
                data.append(entry)
            except (IOError, SyntaxError) as e:
                # 如果无法识别图像文件，打印错误信息并跳过
                print(f"Cannot identify image file {img_path}. Skipping...")
    # 共2887条数据
    print(f"共{len(data)}条数据")
    return data

def save_to_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # 设置images文件夹路径
    images_folder_path = "/root/data/MeiShiTianXia/images"  # 替换为你的images文件夹路径
    output_json_path = "/root/data/MeiShiTianXia/dataset.json"
    
    # 提取信息
    extracted_data = extract_info_from_folder(images_folder_path)
    
    # 保存为JSON文件
    save_to_json(extracted_data, output_json_path)
    
    print(f"数据已成功保存到 {output_json_path}")
