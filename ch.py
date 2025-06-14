import json
import re

def clean_ingredient(ing):
    # Lấy phần trước dấu ':' hoặc '(' hoặc ',' hoặc '-' hoặc chỉ lấy từ đầu đến khi gặp số
    ing = ing.split(":")[0]
    ing = ing.split("(")[0]
    ing = ing.split(",")[0]
    ing = ing.split("-")[0]
    # Loại bỏ số và đơn vị ở cuối
    ing = re.sub(r"\d.*", "", ing)
    return ing.strip().lower()

def process_file(filename):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    for recipe in data:
        if "nguyen_lieu" in recipe:
            # Loại bỏ định lượng, chỉ giữ tên nguyên liệu
            recipe["nguyen_lieu"] = [clean_ingredient(ing) for ing in recipe["nguyen_lieu"]]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_file("mon_an.json")
    process_file("mon_an_2_full.json")