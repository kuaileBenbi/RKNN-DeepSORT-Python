import numpy as np
import cv2
from rknnlite.api import RKNNLite


def initRKNN(rknnModel="rknnModel/yolov8n.rknn"):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    ret = (
        rknn_lite.init_runtime()
    )  # 默认值为 NPU_CORE_AUTO， 即默认使用的是自动调度模式。
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, f"\t\ttrack on NPU")
    return rknn_lite


class Extractor(object):
    def __init__(self, model_path):
        self.size = (64, 128)
        self.rknnPool = initRKNN(model_path)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def _normalize(self, im):
        if im.ndim == 2:
            im = np.stack((im,) * 3, axis=-1)
        # Normalize each channel separately
        im[:, :, 0] = (im[:, :, 0] - self.mean[0]) / self.std[0]
        im[:, :, 1] = (im[:, :, 1] - self.mean[1]) / self.std[1]
        im[:, :, 2] = (im[:, :, 2] - self.mean[2]) / self.std[2]
        return im

    def _resize(self, im, size):
        return cv2.resize(im.astype(np.float32) / 255.0, size)

    def _preprocess(self, im_crops):
        im_batch = np.array(
            [self._normalize(self._resize(im, self.size)) for im in im_crops]
        )
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        sort_results = []
        n = im_batch.shape[0]
        for i in range(n):
            img = (
                im_batch[i].transpose(2, 0, 1).reshape(1, 3, 128, 64)
            )  # Change shape to (1, 3, 128, 64) if channel first
            feature = self.rknnPool.inference(inputs=[img])
            # feature = np.array(feature[0])
            sort_results.append(feature[0])
        return np.concatenate(sort_results, axis=0)


if __name__ == "__main__":
    img = cv2.imread(
        "/home/Tronlong/Desktop/likeai/code/tracker/yolov8-deepsort-py/det_res/0001.jpg"
    )[:, :, (2, 1, 0)]
    extr = Extractor(
        "/home/Tronlong/Desktop/likeai/code/tracker/yolov8-deepsort-py/rknnModel/deepsort.rknn"
    )
    feature = extr([img])
    print(feature.shape)
