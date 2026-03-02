import cv2
import numpy as np
from skimage.metrics import structural_similarity
from metrics import MSE, PSNR, SSIM_windowed, SSIM_windowed_fast 

import matplotlib
matplotlib.use('Agg')  # render without opening a matplotlib window
import matplotlib.pyplot as plt

# ── configuration ─────────────────────────────────────────────
MODE = "video"  # "image" or "video"

image_pairs = [
    ("images/image.jpg", "images/image-MSE-142-SSIM-0662.jpg"),
]

video_folders = {
    "highres":  "video/HighRes",
    "lowres":   "video/LowRes",
    "upscaled": "video/Upscaled",
}

FRAME_PREFIX = "frame_"
FRAME_EXT    = ".png"
FRAME_START  = 2
FRAME_END    = 31
# ──────────────────────────────────────────────────────────────

def put_text(img, text, pos):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

def load_frame(folder, index):
    path = f"{folder}/{FRAME_PREFIX}{index:04d}{FRAME_EXT}"
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")
    return img

def run_image_mode():
    pairs = [
        (cv2.imread(original), cv2.imread(distorted))
        for original, distorted in image_pairs
    ]

    for i, (original, distorted) in enumerate(pairs):
        mse_value = MSE(original, distorted)
        psnr_value = PSNR(original, distorted)
        my_score, my_map = SSIM_windowed(original, distorted)
        sk_score, sk_map = structural_similarity(original, distorted, channel_axis=2, full=True)

        my_map_uint8 = (my_map * 255).astype(np.uint8)
        my_heatmap = cv2.applyColorMap(my_map_uint8, cv2.COLORMAP_JET)

        sk_map_uint8 = (sk_map * 255).astype(np.uint8)
        if sk_map_uint8.ndim == 3:
            sk_map_gray = cv2.cvtColor(sk_map_uint8, cv2.COLOR_BGR2GRAY)
        else:
            sk_map_gray = sk_map_uint8
        sk_heatmap = cv2.applyColorMap(sk_map_gray, cv2.COLORMAP_JET)

        orig_labeled  = original.copy()
        dist_labeled  = distorted.copy()

        put_text(orig_labeled, "Original", (10, 30))
        put_text(dist_labeled, f"MSE: {mse_value:.2f}  PSNR: {psnr_value:.2f}", (10, 30))
        put_text(my_heatmap,   f"My SSIM: {my_score:.4f}", (10, 30))
        put_text(sk_heatmap,   f"skimage SSIM: {sk_score:.4f}", (10, 30))

        combined = np.hstack((orig_labeled, dist_labeled, my_heatmap, sk_heatmap))
        cv2.imshow(f"Pair {i+1}: original | distorted | my SSIM | skimage SSIM", combined)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_video_mode():
    print("Controls:  SPACE = play/pause    A/D = step frame    Q = quit")

    # load all frames
    frames = []
    for idx in range(FRAME_START, FRAME_END + 1):
        hr = load_frame(video_folders["highres"],  idx)
        up = load_frame(video_folders["upscaled"], idx)
        frames.append((idx, hr, up))

    h, w = frames[0][1].shape[:2]

    # panel sizing
    screen_w = 1500
    panel_w = screen_w // 4  # 4 panels: hr, upscaled, ssim heatmap, diff map
    panel_h = int(h * (panel_w / w))

    # precompute all metrics
    print("Precomputing metrics...")
    psnr_values = []
    ssim_values = []
    ssim_maps   = []
    diff_maps   = []

    for idx, hr, up in frames:
        psnr_values.append(PSNR(hr, up))
        score, smap = SSIM_windowed_fast (hr, up)
        ssim_values.append(score)
        ssim_maps.append(smap)

        diff = cv2.absdiff(hr, up)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_maps.append(diff_gray)

    print("Done. Showing results.")

    current = 0
    playing = False

    def make_graph(current_frame):
        fig, axes = plt.subplots(2, 1, figsize=(screen_w / 100, 2.5))
        frame_indices = list(range(FRAME_START, FRAME_END + 1))

        axes[0].plot(frame_indices, psnr_values, color='cyan', linewidth=1.5)
        axes[0].axvline(x=frame_indices[current_frame], color='white', linewidth=1, linestyle='--')
        axes[0].set_ylabel("PSNR", color='white', fontsize=8)
        axes[0].tick_params(colors='white', labelsize=7)
        axes[0].set_facecolor('#1e1e1e')
        axes[0].spines[:].set_color('#444444')

        axes[1].plot(frame_indices, ssim_values, color='lime', linewidth=1.5)
        axes[1].axvline(x=frame_indices[current_frame], color='white', linewidth=1, linestyle='--')
        axes[1].set_ylabel("SSIM", color='white', fontsize=8)
        axes[1].set_xlabel("Frame", color='white', fontsize=8)
        axes[1].tick_params(colors='white', labelsize=7)
        axes[1].set_facecolor('#1e1e1e')
        axes[1].spines[:].set_color('#444444')

        fig.patch.set_facecolor('#1e1e1e')
        fig.tight_layout(pad=0.5)

        # render to numpy array
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        buf = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        plt.close(fig)
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    while True:
        idx, hr, up = frames[current]

        hr_panel = cv2.resize(hr, (panel_w, panel_h))
        up_panel = cv2.resize(up, (panel_w, panel_h))

        # ssim heatmap
        smap_uint8 = (ssim_maps[current] * 255).astype(np.uint8)
        ssim_heatmap = cv2.applyColorMap(smap_uint8, cv2.COLORMAP_JET)
        ssim_panel = cv2.resize(ssim_heatmap, (panel_w, panel_h))

        # diff map
        diff_color = cv2.applyColorMap(diff_maps[current], cv2.COLORMAP_HOT)
        diff_panel = cv2.resize(diff_color, (panel_w, panel_h))

        # labels
        put_text(hr_panel,   f"HighRes  frame {idx:04d}", (10, 30))
        put_text(up_panel,   f"Upscaled  PSNR: {psnr_values[current]:.2f}", (10, 30))
        put_text(ssim_panel, f"SSIM map: {ssim_values[current]:.4f}", (10, 30))
        put_text(diff_panel, "Diff map", (10, 30))

        top_row = np.hstack((hr_panel, up_panel, ssim_panel, diff_panel))

        # graph
        graph = make_graph(current)
        graph_resized = cv2.resize(graph, (top_row.shape[1], 220))

        combined = np.vstack((top_row, graph_resized))
        cv2.imshow("HighRes | Upscaled | SSIM | Diff", combined)

        delay = 33 if playing else 0
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('a'):
            current = max(0, current - 1)
            playing = False
        elif key == ord('d'):
            current = min(len(frames) - 1, current + 1)
            playing = False
        elif playing:
            current = (current + 1) % len(frames)

    cv2.destroyAllWindows()

# ── entry point ───────────────────────────────────────────────
if MODE == "image":
    run_image_mode()
elif MODE == "video":
    run_video_mode()