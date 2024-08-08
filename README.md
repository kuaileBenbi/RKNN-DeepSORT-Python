# RK3588+deepsort 目标检测与跟踪

## python版本使用说明

```
python main.py
```
参数设置见函数：
```

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--VIDEO_PATH", type=str, default="test.mp4")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--save_path", type=str, default="output")
    parser.add_argument("--camera", action="store", dest="cam", type=int, default="-1")
    return parser.parse_args()
```
其他介绍见[RKNN-DeepSORT-CPP](https://github.com/kuaileBenbi/RKNN-DeepSORT-Cpp)