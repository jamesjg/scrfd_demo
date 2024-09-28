# Detect faces in faceCaption5M, write in json, clean data and other tools.




1.detect faces with multi processes and write results in json.

```
python tools/facecaption5m/multi_run.py
```

2.clean json data with score and wirte jsons with different face nums
```
python tools/facecaption5m/clean_facecaption5m_json.py
```
3.count faces with different nums
```
python tools/facecaption5m/count_face_all.py
```
4.reshaple jsons to a new json (facecaption5m_219k)
```
python tools/facecaption5m/resample_facecaption5m.py
```
5.visulize some examples.
```
python tools/facecaption5m/vis_samples.py
```
6.check the path of images in jsons, and can open with cv2 and PIL
```
python tools/facecaption5m/change_facecaption5m_filepath.py
```
7.copy and move selected images to a new folder
```
python tools/facecaption5m/copy_selected_imgs.py
```
8.split the jsons to sub jsons, it is convenient to use multirun
```
python tools/facecaption5m/split_json.py
```