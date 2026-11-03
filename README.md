# CCA 3 – Computer Vision Project

This project uses the two photographs and the video supplied in this chat, together with the uploaded `CCA3_Computer_Vision.ipynb` as the assignment workflow.

## Contents
- `data/original_portrait.png` – supplied formal portrait.
- `data/second_same_scene_photo.jpg` – supplied casual portrait.
- `data/original_video.mp4` – supplied twilight-city video.
- `CCA3_Computer_Vision.ipynb` – notebook with the assignment operations and saved execution results.
- `outputs/task1/` – Gaussian and salt-and-pepper noise, restoration, PSNR/SSIM, Otsu thresholding, morphology, Canny and watershed.
- `outputs/task2/` – HOG visualization, SIFT keypoints/matching, resize experiment and CSV summary.
- `outputs/task3/` – Haar face detection result. YOLO cell is included but not run without Ultralytics and local weights.
- `outputs/task4/` – frame differencing, MOG2, Lucas–Kanade motion comparison and sampled metrics.
- `TECHNICAL_REPORT.md` – actual run notes and limitations.

## Run in Jupyter (Windows / VS Code)
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook
```
Open `CCA3_Computer_Vision.ipynb`, then select **Kernel → Restart Kernel and Run All Cells**.

## Run from terminal
```powershell
python -m pip install -r requirements.txt
jupyter notebook
```
The notebook is the main executable workflow.

## Important input limitations
1. The two photographs are of the same person but in different outfits and settings. They are used for the two-image SIFT comparison, but they are **not photographs of the same scene**. SIFT match results should be interpreted as exploratory.
2. The supplied video is 1920×1080, approximately 25 FPS, and 13.24 seconds long. The assignment notebook asks for a self-recorded 15–30 second video. This supplied clip is shorter and is not verified as self-recorded; replace it with a compliant recording if required.
3. YOLO is included as a notebook cell but was not executed in this package build because `ultralytics` and `yolo11n.pt` weights were not available. Haar face detection was executed.
4. Watershed uses the portrait as a fallback because no dedicated image of multiple/touching objects was supplied. It demonstrates the algorithm but may not be a strong instance-segmentation example.

## Reproducibility
The image-noise experiment uses a fixed random seed (42). CSV metrics and generated visualizations are already included under `outputs/`.


## YOLO object detection (Task 3)

The project includes `src/yolo_detection.py`, which applies pretrained Ultralytics YOLOv8n to:
- `data/original_portrait.png`
- `data/second_same_scene_photo.jpg`
- Up to 8 evenly sampled frames from `data/original_video.mp4`

Run:
```powershell
pip install -r requirements.txt
python src/yolo_detection.py
```
The first run downloads `yolov8n.pt` and therefore needs internet access. Outputs include annotated image/video-frame JPEGs, `outputs/task3/yolo_detections.csv` (bounding boxes, class labels and confidence scores), and `outputs/task3/yolo_class_summary.csv`.

If you are using Jupyter, run the added **Task 3B – YOLO object detection** cells in `CCA3_Computer_Vision.ipynb`.
