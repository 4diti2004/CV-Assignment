"""
CCA 3 Computer Vision practical project.
Run from the project root: python src/run_all.py
"""
from pathlib import Path
import cv2, numpy as np, csv
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)
IMG1 = cv2.imread(str(DATA / "original_portrait_casual.jpg"))
IMG2 = cv2.imread(str(DATA / "second_portrait_formal.jpg"))
if IMG1 is None or IMG2 is None:
    raise FileNotFoundError("Place both portrait images in the data folder using the supplied filenames.")
rng = np.random.default_rng(42)
rgb = cv2.cvtColor(IMG1, cv2.COLOR_BGR2RGB)
h, w = rgb.shape[:2]
gaussian_noisy = np.clip(rgb.astype(np.float32) + rng.normal(0, 22, rgb.shape), 0, 255).astype(np.uint8)
sp_noisy = rgb.copy()
r = rng.random((h, w))
sp_noisy[r < .025] = 0
sp_noisy[r > .975] = 255
gaussian_restored = cv2.GaussianBlur(gaussian_noisy, (5, 5), 1.2)
median_restored = cv2.medianBlur(sp_noisy, 5)

def save_rgb(name, im):
    cv2.imwrite(str(OUT / name), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
save_rgb("01_original.png", rgb)
save_rgb("02_gaussian_noise.png", gaussian_noisy)
save_rgb("03_gaussian_restored.png", gaussian_restored)
save_rgb("04_salt_pepper_noise.png", sp_noisy)
save_rgb("05_median_restored.png", median_restored)
metrics = [
    ["Gaussian noise + Gaussian filter",
     peak_signal_noise_ratio(rgb, gaussian_restored, data_range=255),
     structural_similarity(rgb, gaussian_restored, channel_axis=2, data_range=255)],
    ["Salt-and-pepper noise + Median filter",
     peak_signal_noise_ratio(rgb, median_restored, data_range=255),
     structural_similarity(rgb, median_restored, channel_axis=2, data_range=255)]
]
with open(OUT / "restoration_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f); writer.writerow(["Experiment", "PSNR_dB", "SSIM"]); writer.writerows(metrics)

# Thresholding, morphology, Canny, watershed on second portrait.
gray = cv2.cvtColor(IMG2, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
ops = {
    "06_binary_mask.png": binary,
    "07_erosion.png": cv2.erode(binary, kernel, iterations=1),
    "08_dilation.png": cv2.dilate(binary, kernel, iterations=1),
    "09_opening.png": cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel),
    "10_closing.png": cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel),
    "11_canny_edges.png": cv2.Canny(gray, 80, 160)
}
for name, im in ops.items(): cv2.imwrite(str(OUT / name), im)
sure_bg = cv2.dilate(binary, kernel, iterations=3)
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist, .35 * dist.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)
_, markers = cv2.connectedComponents(sure_fg)
markers += 1
markers[unknown == 255] = 0
markers = cv2.watershed(IMG2.copy(), markers)
watershed = IMG2.copy(); watershed[markers == -1] = [0, 0, 255]
cv2.imwrite(str(OUT / "12_watershed_boundaries.png"), watershed)

# HOG visualization.
gray1 = cv2.cvtColor(IMG1, cv2.COLOR_BGR2GRAY)
_, hog_image = hog(gray1, orientations=9, pixels_per_cell=(16, 16),
                   cells_per_block=(2, 2), visualize=True, block_norm="L2-Hys")
plt.figure(figsize=(7, 8)); plt.imshow(exposure.rescale_intensity(hog_image), cmap="gray")
plt.axis("off"); plt.title("HOG visualization"); plt.tight_layout()
plt.savefig(OUT / "13_hog_visualization.png", dpi=150); plt.close()

# SIFT matching between the two supplied photographs and rotated first image.
def fit_size(im, maxdim=900):
    hh, ww = im.shape[:2]; scale = min(1, maxdim / max(hh, ww))
    return cv2.resize(im, (int(ww * scale), int(hh * scale))) if scale < 1 else im.copy()
a = cv2.cvtColor(fit_size(IMG1), cv2.COLOR_BGR2GRAY)
b = cv2.cvtColor(fit_size(IMG2), cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create(); k1, d1 = sift.detectAndCompute(a, None); k2, d2 = sift.detectAndCompute(b, None)
matcher = cv2.BFMatcher()
def ratio_matches(dA, dB):
    if dA is None or dB is None or len(dB) < 2: return []
    return [m for m, n in matcher.knnMatch(dA, dB, k=2) if m.distance < .75 * n.distance]
matches = ratio_matches(d1, d2)
cv2.imwrite(str(OUT / "14_sift_keypoints_first.jpg"),
            cv2.drawKeypoints(a, k1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.imwrite(str(OUT / "15_sift_keypoints_second.jpg"),
            cv2.drawKeypoints(b, k2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
cv2.imwrite(str(OUT / "16_sift_matching_two_portraits.jpg"),
            cv2.drawMatches(a, k1, b, k2, matches[:60], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))
rot = cv2.warpAffine(a, cv2.getRotationMatrix2D((a.shape[1]/2, a.shape[0]/2), 15, 1),
                     (a.shape[1], a.shape[0]))
kr, dr = sift.detectAndCompute(rot, None); rm = ratio_matches(d1, dr)
cv2.imwrite(str(OUT / "17_sift_rotation_matching.jpg"),
            cv2.drawMatches(a, k1, rot, kr, rm[:60], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS))

# Viola-Jones/Haar face detection.
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_counts = []
for i, im in enumerate([IMG1, IMG2], 1):
    faces = cascade.detectMultiScale(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY),
                                     scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    annotated = im.copy()
    for x, y, ww, hh in faces:
        cv2.rectangle(annotated, (x, y), (x+ww, y+hh), (0,255,0), 3)
        cv2.putText(annotated, "Haar face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), 2)
    cv2.imwrite(str(OUT / f"18_haar_face_detection_{i}.jpg"), annotated)
    face_counts.append(len(faces))

# Video: frame differencing, MOG2 background subtraction and Lucas-Kanade flow.
video = DATA / "sample_video_coverr.mp4"
cap = cv2.VideoCapture(str(video))
if cap.isOpened():
    fps = cap.get(cv2.CAP_PROP_FPS); count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, prev = cap.read()
    if ok:
        prev = cv2.resize(prev, (640, int(prev.shape[0] * 640 / prev.shape[1])))
        prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        mog = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=25, detectShadows=True)
        mog.apply(prev); saved = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 5: continue
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.medianBlur(cv2.threshold(cv2.absdiff(prevgray, gray), 25, 255,
                                                cv2.THRESH_BINARY)[1], 5)
            fg = cv2.medianBlur(mog.apply(frame), 5)
            flowvis = frame.copy()
            p0 = cv2.goodFeaturesToTrack(prevgray, maxCorners=100, qualityLevel=.3,
                                         minDistance=7, blockSize=7)
            if p0 is not None:
                p1, st, err = cv2.calcOpticalFlowPyrLK(prevgray, gray, p0, None)
                if p1 is not None:
                    for new, old in zip(p1[st.flatten()==1], p0[st.flatten()==1]):
                        x, y = new.ravel().astype(int); ox, oy = old.ravel().astype(int)
                        cv2.arrowedLine(flowvis, (ox,oy), (x,y), (0,255,0), 1, tipLength=.3)
                        cv2.circle(flowvis, (x,y), 2, (0,0,255), -1)
            if saved < 6:
                cv2.imwrite(str(OUT / f"video_frame_{saved+1}.jpg"), frame)
                cv2.imwrite(str(OUT / f"video_frame_difference_{saved+1}.jpg"), diff)
                cv2.imwrite(str(OUT / f"video_mog2_foreground_{saved+1}.jpg"), fg)
                cv2.imwrite(str(OUT / f"video_lucas_kanade_{saved+1}.jpg"), flowvis)
                saved += 1
            prevgray = gray
    cap.release()
print("Processing complete. See the outputs folder.")
