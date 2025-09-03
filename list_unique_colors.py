# list_unique_colors.py
from PIL import Image
import numpy as np

def list_unique_colors(image_path, ignore_alpha=True):
    """
    画像内に実際に含まれるRGB値を列挙する．
    :param image_path: 画像パス（PNG推奨）
    :param ignore_alpha: Trueならalpha==0の画素を無視
    :return: [( (R,G,B), count ), ... ] を出現数降順で返す
    """
    im = Image.open(image_path).convert("RGBA")  # Pモードやグレースケールも統一的に扱う
    arr = np.array(im)

    # α=0 を無視（輪郭描画や透過がある場合のノイズ除去に有効）
    if ignore_alpha:
        mask = arr[..., 3] > 0
        rgb = arr[..., :3][mask]
    else:
        rgb = arr[..., :3].reshape(-1, 3)

    # 一意な色とカウント
    colors, counts = np.unique(rgb, axis=0, return_counts=True)

    # 出現数降順にソート
    order = np.argsort(-counts)
    colors, counts = colors[order], counts[order]

    # 結果を整形
    result = [((int(r), int(g), int(b)), int(c)) for (r, g, b), c in zip(colors, counts)]
    return result

if __name__ == "__main__":
    path = "outputs/20250903-195714__segmentation__rayleigh_fd30_snr10__qpsk__hamming74_hdr7xR5_mapPerm12345_ilv16_mtu1024_seed12345/received.png"  # 対象画像
    for (r, g, b), c in list_unique_colors(path):
        print(f"RGB=({r:3d},{g:3d},{b:3d}), HEX=#{r:02X}{g:02X}{b:02X}, count={c}")
