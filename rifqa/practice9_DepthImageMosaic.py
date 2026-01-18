"""
リアルタイム深度測定
"""
import sys
from pathlib import Path
from typing import List, Optional, Tuple
from time import perf_counter

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import settings.settings as settings
from lib.functions import create_filter
from lib.image_util import put_text
from lib import fps

frame_rate = fps.FrameRate()         # 初期化

ROOT_PATH = Path(__file__).resolve().parent
SRC_PATH = ROOT_PATH / "src"
sys.path.append(str(ROOT_PATH))
sys.path.append(str(SRC_PATH))



def main():

    # ===========================================
    # カメラ画像キャプチャ
    # ===========================================
    # capture pipeline
    input_width = 640
    input_height = 480
    framerate = 30
    capture_pipeline = f"v4l2src device=/dev/video0 \
        ! video/x-raw, width=(int){input_width}, height=(int){input_height}, framerate={framerate}/1, format=(string)YUY2 \
        ! videoconvert ! video/x-raw, format=BGR \
        ! appsink max-buffers=1 drop=True"

    # create capture
    cap = cv2.VideoCapture(0)
    try:
        # capture open
        cap.setExceptionMode(True)
        cap.open(capture_pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print(f"{device} is can't opened")
            return

    except Exception as e:
        print(f"{device} is can't opened")
        print("Camera Error", e)
        if cap.isOpened():
            cap.release()
        return

    # =====================================================
    # Writer
    # =====================================================
    # write pipeline
    write_pipeline = f"appsrc ! nveglglessink sync=False"
    # write_pipeline = "appsrc ! video/x-raw, format=BGR ! videoconvert ! x264enc ! flvmux ! filesink location=xyz.flv"

    # create writer
    writer = cv2.VideoWriter()
    try:
        # writer open
        writer.open(write_pipeline, cv2.CAP_GSTREAMER, 0, framerate, (1320, 990))
        if not writer.isOpened():
            print(f"writer is can't opend")
            return

    except Exception as e:
        print(f"writer is can't opened")
        print("Camera Error", e)
        if writer.isOpened():
            writer.release()
        return

    # ===========================================
    # 深度AIモデルの準備
    # ===========================================
    # モデルの読み込み (see https://github.com/intel-isl/MiDaS/#Accuracy for an overview)
    # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy,
    # medium inference speed)
    # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    model_type = "MiDaS_small"

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # モデルをGPUに移動させる（可能であれば
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas = midas.to(device)
    # midas = midas.half()
    midas.eval()    # 推論モード

    # transformsをロードして、大規模または小規模なモデルの画像のサイズを変更および正規化します
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    try:
        while True:
            lap_time = perf_counter()

            ret, image_org = cap.read()
            if image_org is None:
                print("image_org is None")
                continue

            height, width = image_org.shape[:2]
            ratio = settings.CAMERA_IMG_SIZE / height
            img = cv2.resize(
                image_org,
                None,
                fx=ratio,
                fy=ratio,
                interpolation=cv2.INTER_LINEAR)

            if img is None:
                continue
            # ===========================================
            # 深度計算
            # ===========================================
            # 元の解像度を予測してサイズ変更する
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            input_batch = transform(img).to(device,non_blocking=True)

            with torch.no_grad():
                prediction = midas(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            output = prediction.cpu().numpy()

            # ===========================================
            # フィルタ作成
            # ===========================================
            # 近距離、中距離、遠距離
            filter_blur_near, img_filter_blur_near = create_filter(settings.DEPTH_NEAR_MIN, settings.DEPTH_NEAR_MAX, output)
            filter_blur_middle, img_filter_blur_middle = create_filter(settings.DEPTH_MIDDLE_MIN, settings.DEPTH_MIDDLE_MAX, output)
            filter_blur_far, img_filter_blur_far = create_filter(settings.DEPTH_FAR_MIN, settings.DEPTH_FAR_MAX, output)

            # ===========================================
            # 表示画像作成
            # ===========================================
            # 深度画像作成
            sm = plt.cm.ScalarMappable(cmap=None)
            img_depth = sm.to_rgba(output, bytes=True)[:, :, :3]

            # ぼかし画像の作成
            # img_gauss = cv2.GaussianBlur(img, ksize=settings.KARNEL, sigmaX=settings.SIGMAX)
            img_gauss = mosaic(img)

            # カラー変換しておく
            _img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            _img_depth = cv2.cvtColor(img_depth, cv2.COLOR_RGB2BGR)
            _img_gauss = cv2.cvtColor(img_gauss, cv2.COLOR_RGB2BGR)
            _img_filter_blur_near = cv2.cvtColor(img_filter_blur_near, cv2.COLOR_RGB2BGR)
            _img_filter_blur_middle = cv2.cvtColor(img_filter_blur_middle, cv2.COLOR_RGB2BGR)
            _img_filter_blur_far = cv2.cvtColor(img_filter_blur_far, cv2.COLOR_RGB2BGR)

            # 深度パラメータの測定位置に点を書く（画像の真ん中）
            # cv2.circle(_img_depth,
            #            center=(int(_img_depth.shape[1] / 2),
            #                    int(_img_depth.shape[0] / 2)),
            #            radius=5,
            #            color=(255, 255, 255),
            #            thickness=-1)

            # ぼかし画像と入力画像の差分画像作成
            _img_diff = _img_gauss.astype(np.float32) - _img
            # フィルタをかける
            _img_filtered_diff_near = _img_diff * filter_blur_near
            _img_filtered_diff_middle = _img_diff * filter_blur_middle
            _img_filtered_diff_far = _img_diff * filter_blur_far

            # フィルタをかけた後のぼかし画像を足し合わせて、狙った位置だけぼかさない画像生成
            _img_filtered_gauss_near = _img + _img_filtered_diff_near
            _img_filtered_gauss_middle = _img + _img_filtered_diff_middle
            _img_filtered_gauss_far = _img + _img_filtered_diff_far

            # 文字埋め込み
            _img = put_text(_img, "Input")
            _img_gauss = put_text(_img_gauss, "Blur")
            _img_depth = put_text(_img_depth, "depth Image")
            _img_depth = put_text(_img_depth, "MAX:" + str(output.max()), pos=(10, 60))
            _img_depth = put_text(_img_depth, "MIN:" + str(output.min()), pos=(10, 90))
            _img_filter_blur_near = put_text(_img_filter_blur_near, "Near distance filter")
            _img_filter_blur_middle = put_text(_img_filter_blur_middle, "Middle distance filter")
            _img_filter_blur_far = put_text(_img_filter_blur_far, "Long distance filter")
            _img_filtered_gauss_near = put_text(_img_filtered_gauss_near, "Images showing near distance")
            _img_filtered_gauss_middle = put_text(_img_filtered_gauss_middle, "Images showing middle distance")
            _img_filtered_gauss_far = put_text(_img_filtered_gauss_far, "Images showing long distance")

            # 作成した画像をマージする
            img_merge = np.vstack([
                np.hstack([_img, _img_filter_blur_near, _img_filtered_gauss_near.astype(np.uint8)]),
                np.hstack([_img_gauss, _img_filter_blur_middle, _img_filtered_gauss_middle.astype(np.uint8)]),
                np.hstack([_img_depth, _img_filter_blur_far, _img_filtered_gauss_far.astype(np.uint8)])])

if __name__ == "__main__":
    main()