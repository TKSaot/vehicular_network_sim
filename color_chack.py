from PIL import Image
import numpy as np

def get_unique_colors(image_path):
    # 画像をRGBで読み込み
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    
    # ユニークな色を抽出
    unique_colors = np.unique(arr.reshape(-1, 3), axis=0)
    return unique_colors

# 使い方
image_path = "examples/segmentation_00002_.png"  # 自分の画像ファイルパスに変更
colors = get_unique_colors(image_path)

print("色の総数:", len(colors))
print("色リスト (RGB):")
for c in colors:
    print(tuple(c))
