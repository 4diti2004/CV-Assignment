# CCA 3 Computer Vision – Execution Report

## Source materials used
- Uploaded notebook: `CCA3_Computer_Vision.ipynb`
- Uploaded formal portrait: `data/original_portrait.png`
- Uploaded casual portrait: `data/second_same_scene_photo.jpg`
- Uploaded video: `data/original_video.mp4`

The notebook workflow was executed and its outputs are included in this project.

## Unit 3 – Image restoration and segmentation

The first portrait was used as the clean reference. Gaussian noise and salt-and-pepper noise were added with a fixed random seed. Gaussian blur and median filtering were applied.

| Image | PSNR (dB) | SSIM |
|---|---:|---:|
| Gaussian noisy | 24.581 | 0.3027 |
| Gaussian filtered | 30.565 | 0.8183 |
| Salt-and-pepper noisy | 18.078 | 0.4305 |
| Median filtered | 36.373 | 0.9770 |

Otsu thresholding, erosion, dilation, opening, closing and Canny edge detection were also run. Watershed segmentation was run using the portrait as a fallback input because no dedicated touching/multiple-object image was provided.

## Unit 4 – Feature extraction and matching

HOG features and a HOG visualization were generated from the first portrait. SIFT keypoints and descriptors were extracted from both photographs. A second experiment resized image B to 75% and repeated matching.

- **Image A vs image B / fallback pair**: 587 and 1523 keypoints; 22 good matches.
- **Image B vs 75% resized image B**: 1523 and 781 keypoints; 577 good matches.

**Interpretation:** the two supplied photos show the same person but different outfits and backgrounds, so they are not the same scene. Cross-image SIFT matches are exploratory and do not meet a strict requirement for two views of the same scene/object.

## Unit 5 – Object recognition

Viola–Jones Haar Cascade face detection was run on the first portrait and produced **1** face box(es). The annotated image is saved in `outputs/task3/haar_face_detection.jpg`.

YOLO is present as an optional cell in the notebook but was not executed because the `ultralytics` package and local `yolo11n.pt` weights were not available in the execution environment. No YOLO detections are claimed.

## Task 4 – Video motion estimation

The supplied video metadata:
- Resolution: **1920 × 1080**
- Frame rate: **25.00 FPS**
- Frames: **331**
- Duration: **13.24 seconds**

The notebook ran frame differencing, MOG2/GMM background subtraction and Lucas–Kanade optical flow. The method comparison is saved as `outputs/task4/motion_methods_comparison.png`, and sampled measurements are saved as `outputs/task4/motion_sample_metrics.csv`.

**Submission limitation:** the uploaded clip is approximately 13.24 seconds. The assignment notebook specifies a self-recorded video of approximately 15–30 seconds. Replace it with a compliant recording before submission if that requirement is enforced.

## Output files
All generated visualizations and CSV metrics are organized in `outputs/task1` through `outputs/task4`.

## Final review checklist
- [x] Run image restoration and PSNR/SSIM.
- [x] Run thresholding, morphology, Canny and watershed demonstration.
- [x] Run HOG and SIFT experiments.
- [x] Run Haar face detection.
- [x] Run video frame differencing, MOG2 and Lucas–Kanade.
- [ ] Run YOLO after installing Ultralytics and supplying `yolo11n.pt`.
- [ ] Replace the supplied video with a self-recorded 15–30 second clip if required.
- [ ] Supply two images of the same scene/object if required for strict SIFT task compliance.


## YOLO implementation status and reproducibility

A YOLOv8n inference script has been added at `src/yolo_detection.py` and a matching execution section has been added to the notebook. It processes both supplied photographs and up to eight evenly sampled frames from the supplied video, then writes annotated images, per-detection class/confidence/bounding-box CSV data and a class-count summary under `outputs/task3/`.

**Execution status for this packaged run:** YOLO inference could not be executed in the current build environment because the `ultralytics` package and pretrained weights were not present, and package/model download access was unavailable. Therefore no YOLO detections or YOLO-vs-Haar numerical comparison are claimed here. To generate genuine results on your Windows/Jupyter computer, run `pip install -r requirements.txt` and then `python src/yolo_detection.py` (or run the added notebook cells) while connected to the internet. The script downloads `yolov8n.pt` on first use. Use the generated `yolo_detections.csv` and annotated images for the YOLO-vs-Haar discussion after execution.
