import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# Đường dẫn ảnh gốc
input_folder = r"C:\Users\admin\Desktop\venvpython\Nguyen_lieu"

# Đường dẫn để lưu ảnh sau khi xử lý
output_folder = 'processed_images'
os.makedirs(output_folder, exist_ok=True)

# Đóng gói một số phép augment bằng albumentations
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])

processed_images = []

# Duyệt qua tất cả ảnh trong folder
for file_name in os.listdir(input_folder):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
        img_path = os.path.join(input_folder, file_name)

        # Đọc ảnh
        with open(img_path, 'rb') as f:
            img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Lỗi đọc ảnh: {img_path}")
            continue

        # Chuyển sang RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize về 640x640
        resized = cv2.resize(img, (640, 640))

        # Tăng cường ảnh
        aug = transform(image=resized)['image']

        # Chuẩn hoá về [0,1]
        normalized = aug / 255.0

        processed = (normalized * 255).astype(np.uint8)
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)

        output_file = os.path.join(output_folder, file_name)
        cv2.imwrite(output_file, processed)

        processed_images.append(processed)

print("Hoàn tất xử lý ảnh!")