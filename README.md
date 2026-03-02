## Image Quality Report

A Python tool for evaluating image and video upscaling quality using PSNR and SSIM metrics.
Built as a practical exploration of quality assessment techniques
---
## Purpose

Evaluating how close an upscaled frame is to the original high-resolution reference is a core part of validating that training data and model output.

## Features

- Per-image mode: compare a reference and distorted image with MSE, PSNR, SSIM score, SSIM heatmap
- Per-video mode: step through frame sequences comparing HighRes vs Upscaled with live metric graphs
- SSIM implemented from scratch using a Gaussian-weighted sliding window based on Wang et al.
- Diff map and SSIM heatmap visualizations for spatial quality investigation
- Frame-by-frame PSNR and SSIM graphs with current frame indicator
---
dataset for the videos, city: https://people.tuebingen.mpg.de/msajjadi/FRVSR_Vid4.zip
---
paper: WHY IS IMAGE QUALITY ASSESSMENT SO DIFFICULT?
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ece.uwaterloo.ca/~z70wang/publications/icassp02a.pdf

SSIM paper: "On the Mathematical Properties of the Structural Similarity Index"
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://ece.uwaterloo.ca/~z70wang/publications/TIP_SSIM_MathProperties.pdf

PSNR and other measurements: "Information Content Weighting for Perceptual Image Quality Assessment"
chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://www.ece.uwaterloo.ca/~z70wang/publications/IWSSIM.pdf

---
<img width="864" height="554" alt="image" src="https://github.com/user-attachments/assets/e1237609-fa0e-4f2a-9d3b-925cfd5e434f" />
<img width="409" height="206" alt="image" src="https://github.com/user-attachments/assets/0a44e359-19db-4638-9696-a35433132363" />



---

since this old mnetrics are getting old, more advanced metrics are being made by companies to try and deal with, ghosting, temporal flicker, shimmering noise, and even hallucinated textures
https://community.intel.com/t5/Blogs/Tech-Innovation/Client/Assessing-Video-Quality-in-Real-time-Computer-Graphics/post/1694109






