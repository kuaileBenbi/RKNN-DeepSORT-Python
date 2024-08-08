from datetime import datetime
import cv2
import time
import argparse
import os
import json

from DeepSORT.deep_sort import DeepSort
from YOlOv8.func import draw, expand_bbox_xyxy, myFunc
from YOlOv8.detector import detectExecutor
from utils.draw import draw_boxes
from utils.io import write_results
from utils.log import get_logger


class VideoTracker(object):
    def __init__(self, args, video_path):
        self.args = args
        self.video_path = video_path
        self.TPEs = 2
        self.logger = get_logger("root")

        if args.display:
            cv2.namedWindow("show", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("show", 800, 600)

        if args.cam != -1:
            print("Using webcam " + str(args.cam))
            self.vdo = cv2.VideoCapture(args.cam)
        else:
            self.vdo = cv2.VideoCapture()

        # initialize detector and tracker
        self.detector = detectExecutor(
            det_model="rknnModel/yolov8s.rknn", TPEs=2, func=myFunc
        )
        self.deepsort = DeepSort(
            extractor_model="rknnModel/deepsort.rknn",
            max_dist=0.2,
            min_confidence=0.5,
            nms_max_overlap=0.5,
            max_iou_distance=0.7,
            max_age=70,
            n_init=3,
            nn_budget=100,
        )

    def __enter__(self):
        if self.args.cam != -1:
            ret, frame = self.vdo.read()
            assert ret, "Error: Camera error"
            self.im_width = frame.shape[0]
            self.im_height = frame.shape[1]

        else:
            assert os.path.isfile(self.video_path), "Path error"
            self.vdo.open(self.video_path)
            self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert self.vdo.isOpened()

        # TODO save masks
        if self.args.save_path:
            os.makedirs(self.args.save_path, exist_ok=True)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_video_path = os.path.join(
                self.args.save_path, f"results_{current_time}.mp4"
            )
            self.save_results_path = os.path.join(
                self.args.save_path, f"results_{current_time}.txt"
            )

            # create video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(
                self.save_video_path, fourcc, 25, (self.im_width, self.im_height)
            )
            # logging
            self.logger.info("Save results to {}".format(self.args.save_path))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        track_results_txt = []
        idx_frame = 0
        with open("coco_classes.json", "r") as f:
            idx_to_class = json.load(f)

        # Initialize the frames required for async
        if self.vdo.grab():
            for i in range(self.TPEs + 1):
                ret, im = self.vdo.retrieve()
                if not ret:
                    del pool
                    exit(-1)
                self.detector.put(im)

        while self.vdo.grab():
            idx_frame += 1
            if idx_frame % self.args.frame_interval:
                continue

            start = time.time()
            ret, im = self.vdo.retrieve()
            if not ret:
                break
            self.detector.put(im)
            # do detection
            # draw_frame, flag = self.detector.get()
            (cur_frame, results), flag = self.detector.get()
            if flag == False:
                break
            ltbr_boxes = results["ltbr_boxes"]
            cls_ids = results["classes"]
            cls_conf = results["scores"]

            # ==========draw bbox&label==============#
            # 1.对应detector输出bbox
            # if ltbr_boxes is not None:
            #     draw(cur_frame, ltbr_boxes, cls_conf, cls_ids)
            #     filename = f"{idx_frame:04d}.jpg"
            #     cv2.imwrite(filename, cur_frame)
            # else:
            #     print(f"{idx_frame} detected failed!")
            # 2.对应detecto输出frame
            # filename = f"{idx_frame:04d}.jpg"
            # cv2.imwrite(filename, draw_frame)
            # ==========draw bbox&label==============#

            # ==========select special class=========#
            # mask = cls_ids == 0    # 0 is the person
            # ltbr_boxes = ltbr_boxes[mask]
            # cls_conf = cls_conf[mask]
            # cls_ids = cls_ids[mask]
            # ==========select special class=========#

            # do tracking
            if ltbr_boxes is not None:
                # print("========================================")
                # print(f"Tracking frame_{idx_frame}...")
                # bbox dilation just in case bbox too small, delete this line if using a better pedestrian detector
                bbox_cxcywh = expand_bbox_xyxy(ltbr_boxes)
                outputs, _ = self.deepsort.update(
                    bbox_cxcywh, cls_conf, cls_ids, cur_frame
                )
                # print(f"Track Done!")

                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_tlwh = []
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    cls = outputs[:, -2]
                    names = [idx_to_class[str(label)] for label in cls]

                    cur_frame = draw_boxes(cur_frame, bbox_xyxy, names, identities)

                    for bb_xyxy in bbox_xyxy:
                        bbox_tlwh.append(self.deepsort._xyxy_to_tlwh(bb_xyxy))

                    track_results_txt.append(
                        (idx_frame - 1, bbox_tlwh, identities, cls)
                    )

                end = time.time()

                if self.args.display:
                    cv2.imshow("show", cur_frame)
                    cv2.waitKey(1)

                if self.args.save_path:
                    self.writer.write(cur_frame)
                    # filename = f"{idx_frame:04d}.jpg"
                    # cv2.imwrite(filename, cur_frame)

                write_results(self.save_results_path, track_results_txt, "mot")

                # logging
                self.logger.info(
                    f"time: {end - start:.03f}s, fps: {1 / (end - start):.03f}, detection numbers: {ltbr_boxes.shape[0]}, tracking numbers: {len(outputs)}"
                )

        while True:
            idx_frame += 1
            (cur_frame, results), flag = self.detector.get()
            if flag == False:
                break
            ltbr_boxes = results["ltbr_boxes"]
            cls_ids = results["classes"]
            cls_conf = results["scores"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="test.mp4")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with VideoTracker(args, video_path=args.VIDEO_PATH) as vdo_trk:
        vdo_trk.run()
