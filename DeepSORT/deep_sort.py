import numpy as np


from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

# visualize
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import seaborn as sns

__all__ = ["DeepSort"]


class DeepSort(object):
    def __init__(
        self,
        extractor_model,
        max_dist=0.2,
        min_confidence=0.3,
        nms_max_overlap=1.0,
        max_iou_distance=0.7,
        max_age=70,
        n_init=3,
        nn_budget=100,
    ):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap
        self.extractor = Extractor(extractor_model)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init
        )

    def update(self, bbox_xywh, confidences, classes, ori_img, masks=None):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)

        # if len(features) > 0:
        #     # ======visualize======== #
        #     labels = np.array(classes)
        #     self.visualize_features(features, labels)
        #     # ======visualize======== #

        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        """过滤掉低置信度检测结果"""
        detections = [
            Detection(
                bbox_tlwh[i],
                conf,
                label,
                features[i],
                None if masks is None else masks[i],
            )
            for i, (conf, label) in enumerate(zip(confidences, classes))
            if conf > self.min_confidence
        ]  # 只有置信度大于或等于这个值的检测结果才会被考虑用于跟踪

        # run on non-maximum supression
        """非极大值抑制 移除重叠目标"""
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(
            boxes, self.nms_max_overlap, scores
        )  # 在执行NMS时，如果两个检测框的IoU（交并比）超过这个值，则认为它们是重叠的，需要抑制其中一个
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        mask_outputs = []
        for track in self.tracker.tracks:
            """只输出有把握的跟踪目标"""
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            track_cls = track.cls
            outputs.append(
                np.array([x1, y1, x2, y2, track_cls, track_id], dtype=np.int32)
            )
            if track.mask is not None:
                mask_outputs.append(track.mask)
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs, mask_outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        else:
            exit(-1)
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.0
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.0
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    @staticmethod
    def _xyxy_to_tlwh(bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    # def visualize_features(self, features, labels):
    #     """
    #     Visualize features using PCA and t-SNE.

    #     Parameters:
    #     - features: numpy array of shape (n_samples, n_features)
    #     - labels: numpy array of shape (n_samples,) containing the labels for each feature
    #     """
    #     n_samples, n_features = features.shape  # if n_samples==1: n_features=512
    #     # 创建图形窗口
    #     fig, axs = plt.subplots(
    #         3, max(n_samples, 3), figsize=(max(n_samples, 3) * 5, 15)
    #     )

    #     # 直方图：显示特征的分布
    #     axs[0, 0].hist(features.ravel(), bins=50, alpha=0.7)
    #     axs[0, 0].set_title("Feature Distribution Histogram")
    #     # 使用PCA进行降维
    #     if n_samples > 1 and n_features > 1:
    #         pca = PCA(n_components=2)
    #         features_pca = pca.fit_transform(features)

    #         # 可视化PCA降维结果
    #         for label in np.unique(labels):
    #             axs[0, 1].scatter(
    #                 features_pca[labels == label, 0],
    #                 features_pca[labels == label, 1],
    #                 label=f"Target {label}",
    #                 alpha=0.5,
    #             )
    #         axs[0, 1].set_title("PCA of Features")
    #         axs[0, 1].legend()

    #         # 使用t-SNE进行降维
    #         tsne = TSNE(n_components=2, random_state=42)
    #         features_tsne = tsne.fit_transform(features)

    #         # 可视化t-SNE降维结果
    #         for label in np.unique(labels):
    #             axs[1, 0].scatter(
    #                 features_tsne[labels == label, 0],
    #                 features_tsne[labels == label, 1],
    #                 label=f"Target {label}",
    #                 alpha=0.5,
    #             )
    #         axs[1, 0].set_title("t-SNE of Features")
    #         axs[1, 0].legend()
    #     else:
    #         axs[0, 1].text(
    #             0.5,
    #             0.5,
    #             "Not enough samples or features to perform PCA",
    #             ha="center",
    #             va="center",
    #         )
    #         axs[1, 0].text(
    #             0.5,
    #             0.5,
    #             "Not enough samples or features to perform t-SNE",
    #             ha="center",
    #             va="center",
    #         )

    #     feature_map_shape = (32, 16)

    #     for i in range(n_samples):  # 仅可视化前3个样本的特征图
    #         feature_map = features[i].reshape(feature_map_shape)
    #         sns.heatmap(feature_map, ax=axs[2, i], cmap="viridis")
    #         axs[2, i].set_title(f"Feature Map {i+1}")

    #     plt.tight_layout()
    #     plt.savefig(
    #         "/Users/lixiwang/Documents/projects/tracker/code/deep_sort_pytorch/output/current_frame_features.png"
    #     )
    #     plt.close()
