import os
import json

folder_path = './result'

files = os.listdir(folder_path)

arr = []


for file in files:
    if file.endswith('.json'):
        full_path = os.path.join(folder_path, file)
        try:
            with open(full_path,'r', encoding='utf-8') as text:
                data = json.load(text)
                arr.append(data.get('refined_text', "Null"))
        except KeyError:
            print('Такого ключа нет')

with open("data.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(arr))
