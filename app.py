import streamlit as st
import numpy as np
import cv2
import tempfile
import json
import unidecode
from inference_sdk import InferenceHTTPClient

st.title("Nhận dạng nguyên liệu & Gợi ý món ăn")

# Đọc dữ liệu món ăn từ file
def load_recipes():
    recipes = []
    try:
        with open("20_mon_chay_dau.json", encoding="utf-8") as f:
            recipes += json.load(f)
    except Exception:
        pass
    # Chuẩn hóa nguyên liệu cho từng món
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

        # Nhận dạng nguyên liệu bằng inference_sdk
        CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="zFi7uXEd69xEQlaRyoqd"
)
        result = CLIENT.infer(temp_image_path, model_id="monchay/3")

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

        # Thống kê tên đối tượng nhận dạng
        st.subheader("Các nguyên liệu nhận dạng được:")
        ingredient_names = set()
        for prediction in result.get("predictions", []):
            label = prediction["class"]
            ingredient_names.add(label)
        if ingredient_names:
            st.write(", ".join(ingredient_names))
        else:
            st.write("Không nhận dạng được nguyên liệu nào.")

        # Gợi ý món ăn phù hợp
        st.subheader("Gợi ý món ăn phù hợp:")
        suggested_recipes = []
        if detected_ingredients:
            for recipe in recipes:
                recipe_ingredients = set(recipe.get("nguyen_lieu_norm", []))
                matched = len(recipe_ingredients & detected_ingredients)
                # Phần lớn nghĩa là >= 50% số nguyên liệu của món ăn đó
                required = max(1, int(np.ceil(len(recipe_ingredients) / 2)))
                if matched >= required:
                    suggested_recipes.append((recipe, matched / len(recipe_ingredients)))
            suggested_recipes = sorted(suggested_recipes, key=lambda x: x[1], reverse=True)[:3]
        if suggested_recipes:
            for recipe, _ in suggested_recipes:
                st.markdown(f"### {recipe.get('ten_mon', 'Không rõ tên món')}")
                st.write(f"**Nguyên liệu:** {', '.join(recipe.get('nguyen_lieu', [])) if recipe.get('nguyen_lieu') else 'Không có thông tin'}")
                st.write("---")
        else:
            st.write("Không tìm thấy món ăn phù hợp với nguyên liệu nhận dạng được.")

        st.subheader("Kết quả JSON trả về:")
        st.json(result)

    except Exception as e:
        st.error(f"Đã xảy ra lỗi trong quá trình xử lý: {e}")
