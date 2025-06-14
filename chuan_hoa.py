import json
import unidecode

def normalize_ingredient(ing):
    return unidecode.unidecode(ing.lower())

def process_file(filename):
    with open(filename, encoding="utf-8") as f:
        data = json.load(f)
    for recipe in data:
        if "nguyen_lieu" in recipe:
            recipe["nguyen_lieu"] = [normalize_ingredient(ing) for ing in recipe["nguyen_lieu"]]
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    process_file("mon_an.json")
    process_file("mon_an_2_full.json")