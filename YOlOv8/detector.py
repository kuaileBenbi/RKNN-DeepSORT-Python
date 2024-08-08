from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed


def initRKNN(rknnModel="rknnModel/yolov8n.rknn", id=0):
    rknn_lite = RKNNLite()
    ret = rknn_lite.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    if id == 0:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn_lite.init_runtime()
    if ret != 0:
        print("Init runtime environment failed")
        exit(ret)
    print(rknnModel, f"\t\tdetect on NPU-{id}")
    return rknn_lite


def initRKNNs(rknnModel="rknnModel/yolov8n.rknn", TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list


class detectExecutor:
    def __init__(self, det_model, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.rknnPool = initRKNNs(det_model, TPEs)
        self.detPool = ThreadPoolExecutor(max_workers=TPEs)
        self.func = func
        self.num = 0

    def put(self, frame):
        self.queue.put(
            self.detPool.submit(self.func, self.rknnPool[self.num % self.TPEs], frame)
        )
        self.num += 1

    def get(self):
        if self.queue.empty():
            return (None, None), False
        fut = self.queue.get()
        return fut.result(), True

    def release(self):
        self.detPool.shutdown()
        for rknn_lite in self.rknnPool:
            rknn_lite.release()
