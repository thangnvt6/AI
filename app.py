import streamlit as st
import numpy as np
import cv2
import tempfile
import json
import unidecode
import requests

st.title("Nhận dạng hình ảnh & Gợi ý món ăn")

# Đọc dữ liệu món ăn từ 2 file
def load_recipes():
    recipes = []
    try:
        with open("mon_an.json", encoding="utf-8") as f1:
            recipes += json.load(f1)
    except Exception:
        pass
    try:
        with open("mon_an_2_full.json", encoding="utf-8") as f2:
            recipes += json.load(f2)
    except Exception:
        pass
    # Chuẩn hóa nguyên liệu cho từng món (không có gạch dưới)
    for recipe in recipes:
        if "nguyen_lieu" in recipe:
            recipe["nguyen_lieu_norm"] = [unidecode.unidecode(ing.lower()) for ing in recipe["nguyen_lieu"]]
        else:
            recipe["nguyen_lieu_norm"] = []
    return recipes

recipes = load_recipes()

uploaded_file = st.file_uploader("Chọn một hình ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, 1)
    st.image(original_image, channels="BGR", caption="Hình ảnh gốc", use_container_width=True)
    st.write("Đang xử lý hình ảnh...")

    try:
        # Lưu ảnh tạm thời
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(cv2.imencode('.jpg', original_image)[1].tobytes()) 
            temp_image_path = tmp_file.name

        # Gửi ảnh đến API Roboflow bằng requests
        url = "https://serverless.roboflow.com/vegetables-detection-ryup0-b08kl/1"
        api_key = "zFi7uXEd69xEQlaRyoqd"

        with open(temp_image_path, "rb") as img_file:
            files = {"file": img_file}
            params = {"api_key": api_key}
            response = requests.post(url, files=files, params=params)

        if response.status_code == 200:
            result = response.json()
        else:
            st.error(f"API trả về lỗi {response.status_code}: {response.text}")
            result = {"predictions": []}

        # Vẽ bounding box và nhãn lên ảnh
        image_annotated = original_image.copy()
        detected_ingredients = set()
        for prediction in result.get("predictions", []):
            x, y, w, h = prediction["x"], prediction["y"], prediction["width"], prediction["height"]
            label = prediction["class"]
            detected_ingredients.add(unidecode.unidecode(label.lower()))
            # Tính tọa độ góc trên trái và dưới phải
            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)
            # Vẽ hình chữ nhật
            cv2.rectangle(image_annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Vẽ nhãn
            cv2.putText(image_annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        st.subheader("Hình ảnh đã gán nhãn:")
        st.image(image_annotated, channels="BGR", use_container_width=True)

        # Thống kê tên và số lượng đối tượng
        st.subheader("Thống kê đối tượng nhận dạng:")
        object_counts = {}
        for prediction in result.get("predictions", []):
            label = prediction["class"]
            object_counts[label] = object_counts.get(label, 0) + 1
        if object_counts:
            for label, count in object_counts.items():
                st.write(f"- {label}: {count}")
        else:
            st.write("Không nhận dạng được đối tượng nào.")
        
        # Gợi ý món ăn phù hợp
        st.subheader("Gợi ý món ăn phù hợp:")
        suggested_recipes = []
        if len(detected_ingredients) >= 2:
            for recipe in recipes:
                if all(ingredient in recipe.get("nguyen_lieu_norm", []) for ingredient in detected_ingredients):
                    suggested_recipes.append(recipe)
        else:
            for recipe in recipes:
                if any(ingredient in recipe.get("nguyen_lieu_norm", []) for ingredient in detected_ingredients):
                    suggested_recipes.append(recipe)

        if suggested_recipes:
            for recipe in suggested_recipes:
                st.markdown(f"### {recipe.get('ten_mon', 'Không rõ tên món')}")
                if recipe.get("img_url"):
                    st.image(recipe["img_url"], caption="Ảnh minh họa", use_container_width=True)
                if recipe.get("youtube_url"):
                    st.markdown(f"[Xem video hướng dẫn]({recipe['youtube_url']})")
                st.write(f"**Nguyên liệu:** {', '.join(recipe.get('nguyen_lieu', [])) if recipe.get('nguyen_lieu') else 'Không có thông tin'}")
                st.write("---")
        else:
            st.write("Không tìm thấy món ăn phù hợp với nguyên liệu nhận dạng được.")
        
        st.subheader("Kết quả JSON trả về:")
        st.json(result)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
