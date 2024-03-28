# %%
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_point_cloud(depth_npz_path, color_png_path):
    # 深度画像とカラー画像の読み込み
    depth_data = np.load(depth_npz_path)['depth']  # 'arr_0'はデフォルトの変数名です

    height, width = depth_data.shape
    color_image = Image.open(color_png_path).resize(
        (width, height))
    color_data = np.asarray(color_image)

    # 点群の生成
    points = []
    for y in range(height):
        for x in range(width):
            # 深度値に基づいて3次元座標を計算
            depth = depth_data[y, x]
            point = [x, y, depth]
            # カラー値を取得
            color = color_data[y, x]
            # 点群に追加
            points.append(point + color.tolist())

    return np.array(points)


def plot_point_cloud(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 点群の座標を分解
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    # 点群の色情報を取得（RGB）
    colors = points[:, 3:6] / 255  # matplotlibでは色の値が0から1の間でなければならない

    # 点群をプロット
    ax.scatter(xs, ys, zs, color=colors, marker='.')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


# %%
# 例: 'depth.npz'と'image.png'から点群を生成
point_cloud = create_point_cloud('./runs/train/default_config-train_PxCam-2024.03.26-09h30m39s/output_00000033.npz',
                                 '/home/miyata/dataset/model_of_mouth/WIN_20240325_18_01_15_Pro.mp4/00000033.png')


# %%
# 例: 点群をプロット
plot_point_cloud(point_cloud)


def save_point_cloud_as_obj(points, file_name):
    with open(file_name, 'w') as file:
        for point in points:
            # 座標データの書き出し
            file.write(f'v {point[0]} {point[1]} {point[2]}\n')


# 点群データをOBJファイルとして保存
save_point_cloud_as_obj(point_cloud, 'point_cloud.obj')

# %%
