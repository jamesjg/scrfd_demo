## Installation
1. Install [mmcv](https://mmcv.readthedocs.io/en/v1.4.8/get_started/installation.html) and [torch](https://pytorch.org/get-started/previous-versions/). (mmcv-full==1.4.8 with pytorch==1.11.0 was tested)
```
    pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

    pip install mmdet

```

2. Install build requirements.
```
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
```

3. Install onnxruntime
```
    pip install onnxruntime-gpu 
    pip install onnx
```

known issue: if you encounter the error: `gcc: error: ../common/maskApi.c: 没有那个文件或目录`, the reason is that the version of Cython is too high, you can downgrade Cython to 0.29.21 by running :
```
pip3 install cython==0.29.33 -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

## Pretrained-Models

|      Name      | Easy  | Medium | Hard  | FLOPs | Params(M) | Infer(ms) | Link                                                         |
| :------------: | ----- | ------ | ----- | ----- | --------- | --------- | ------------------------------------------------------------ |
|   SCRFD_500M   | 90.57 | 88.12  | 68.51 | 500M  | 0.57      | 3.6       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyYWxScdiTITY4TQ?e=DjXof9) |
|    SCRFD_1G    | 92.38 | 90.57  | 74.80 | 1G    | 0.64      | 4.1       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyPVLI44ahNBsOMR?e=esPrBL) |
|   SCRFD_2.5G   | 93.78 | 92.16  | 77.87 | 2.5G  | 0.67      | 4.2       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyTIXnzB1ujPq4th?e=5t1VNv) |
|   SCRFD_10G    | 95.16 | 93.87  | 83.05 | 10G   | 3.86      | 4.9       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyUKwTiwXv2kaa8o?e=umfepO) |
|   SCRFD_34G    | 96.06 | 94.92  | 85.29 | 34G   | 9.80      | 11.7      | [download](https://1drv.ms/u/s!AswpsDO2toNKqyKZwFebVlmlOvzz?e=V2rqUy) |
| SCRFD_500M_KPS | 90.97 | 88.44  | 69.49 | 500M  | 0.57      | 3.6      | [download](https://1drv.ms/u/s!AswpsDO2toNKri_NDM0GIkPpkE2f?e=JkebJo) |
| SCRFD_2.5G_KPS | 93.80 | 92.02  | 77.13 | 2.5G  | 0.82      | 4.3       | [download](https://1drv.ms/u/s!AswpsDO2toNKqyGlhxnCg3smyQqX?e=A6Hufm) |
| SCRFD_10G_KPS  | 95.40 | 94.01  | 82.80 | 10G   | 4.23      | 5.0       | [download](https://1drv.ms/u/s!AswpsDO2toNKqycsF19UbaCWaLWx?e=F6i5Vm) |

mAP, FLOPs and inference latency are all evaluated on VGA resolution.
``_KPS`` means the model includes 5 keypoints prediction.

Download the model and put it in the `work_dirs/$MODEL_NAME$/` directory.

## Convert model to onnx
```
python tools/scrfd2onnx.py configs/scrfd/scrfd_10g_bnkps.py work_dirs/scrfd_10g_bnkps/model.pth --input-img t1.jpg 
```

You can also set specific input shape by pass --shape h w, then output onnx model can be optimized by onnx-simplifier.

The result onnx model will be saved in the `onnx/` directory.

## Inference with onnx model
```
 python tools/scrfd.py  onnx/scrfd_10g_bnkps.onnx --input-img trump.jpg
```

The result image will be saved in the `outputs/` directory.

## Download onnx model
|      Name      | Link                                                         |
| :------------: | ------------------------------------------------------------ |
| SCRFD_10G_KPS(640*640)  | [download](https://pan.baidu.com/s/1q_8lIRjos7PrGjauE1lKpg?pwd=qmhm) 提取码:qmhm |