# Skin Cancer AI Risk Assessment System

**Based on ConvNeXt-Small + Grad-CAM**

Last updated: 2026
Author: Yixuan Huang. Developed with AI assistance; the training and deployment pipeline is fully self-contained.

---

## Contents

1. Project overview
2. Files
3. System requirements
4. Installing dependencies
5. Preparing the dataset
6. Training the model (`train.py`)
7. Command-line inference (`infer.py`)
8. Web interface (`app.py`)
9. Output files
10. Troubleshooting
11. Important notes

---

## 1. Project overview

This project is a complete AI system for classifying skin lesions and assessing risk. It analyses dermoscopy images, outputs the probability of each of three classes, and produces Grad-CAM heatmaps that show which regions of the image the model focused on.

**Classes**

- **Melanoma** (`0_melanoma`, shown as 黑色素瘤): the most dangerous type of skin cancer
- **Other skin cancer** (`1_other_cancer`, shown as 皮肤癌): other malignant skin tumours
- **Non-cancerous / healthy** (`2_other_conditions`, shown as 非癌症/健康): benign lesions or normal skin

**Core techniques**

- **Model architecture:** ConvNeXt-Small (Meta AI, 2022)
- **Heatmaps:** Grad-CAM (gradient-weighted class activation mapping)
- **Loss function:** HybridLoss (0.4 × cross-entropy + 0.6 × focal loss)
- **Weights:** EMA (exponential moving average)
- **Mixed precision:** AMP (automatic mixed precision, for faster GPU training)
- **Front end:** Flask + plain HTML/CSS/JavaScript

**Results** (ISIC 2018, validation set of 2,441 images, best epoch)

| Metric | Value |
| --- | --- |
| Macro F1 | 0.88 |
| Macro AUC | 0.95 |
| Accuracy | 94% |

---

## 2. Files

Project folder structure (all files in the same directory):

```
your-project-folder/
│
├── train.py              ← [Core] model training script
├── infer.py              ← command-line inference script (supports TTA)
├── app.py                ← web interface (Flask server)
├── README.md             ← this file
│
├── best_model.pth        ← generated after training (best model weights)
├── checkpoint_latest.pth ← generated after training (for resuming training)
├── training_log.csv      ← generated after training (metrics for each epoch)
│
├── skin_cancer_data/     ← [Provide your own] training dataset
│   ├── train/
│   │   ├── 0_melanoma/
│   │   ├── 1_other_cancer/
│   │   └── 2_other_conditions/
│   └── val/
│       ├── 0_melanoma/
│       ├── 1_other_cancer/
│       └── 2_other_conditions/
│
├── orgimg/               ← [Web input] put images to analyse here
├── processedimg/         ← [Web output] analysis results are generated here
│
└── reports/              ← generated during training (evaluation charts)
    ├── confusion_matrix_best.png
    ├── roc_curves_best.png
    └── classification_report_best.txt
```

---

## 3. System requirements

**Minimum (inference / web interface only)**

- Python 3.9 or later
- 8 GB RAM
- 2 GB disk space (model + libraries)
- GPU optional (runs on CPU, but more slowly)

**Recommended (training)**

- Python 3.10 / 3.11
- NVIDIA RTX GPU with 8 GB VRAM or more
- CUDA 11.8 or later
- 16 GB RAM

**Tested environment**

- Windows 11 + Anaconda Python 3.11 + RTX 4060 Laptop GPU

---

## 4. Installing dependencies

Open Command Prompt (CMD) and run the following commands in order.

**Step 1: Install PyTorch (GPU version, CUDA 11.8)**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

If your computer has no dedicated GPU, install the CPU version instead:

```bash
pip install torch torchvision
```

**Step 2: Install training dependencies**

```bash
pip install scikit-learn tensorboard seaborn matplotlib
```

**Step 3: Install front-end dependencies (note the NumPy version)**

```bash
pip install "numpy==1.26.4"
pip install "opencv-python-headless==4.9.0.80"
pip install flask
```

**Step 4: Check the installation**

```bash
python -c "import torch, cv2, flask, sklearn; print('All dependencies installed')"
```

---

## 5. Preparing the dataset

**① Create this folder structure in the project folder:**

```
skin_cancer_data/
  train/
    0_melanoma/          ← melanoma training images (500+ recommended)
    1_other_cancer/      ← other skin cancer training images (500+ recommended)
    2_other_conditions/  ← non-cancerous images (500+ recommended)
  val/
    0_melanoma/          ← melanoma validation images (100+ recommended)
    1_other_cancer/      ← other skin cancer validation images (100+ recommended)
    2_other_conditions/  ← non-cancerous validation images (100+ recommended)
```

**② Image requirements**

- Supported formats: `.jpg` / `.jpeg` / `.png`
- Recommended resolution: 224 × 224 or higher (images are resized automatically)
- Each class needs at least 20 images to pass the data check

**③ Train / validation split**

- Recommended ratio: 8:2 or 7:3
- Try to keep the classes balanced; if they are not, class weights are applied automatically

**④ Public datasets**

- This project was trained on **ISIC 2018** (dermoscopy images), with the original seven diagnoses grouped into the three classes above.
- Other suitable datasets: ISIC 2019/2020, HAM10000, Kaggle Skin Cancer MNIST

---

## 6. Training the model (`train.py`)

**Basic usage**

```bash
python train.py
```

**Training process**

*Phase 1 (10 epochs)*

- The ConvNeXt-S backbone is fully frozen
- Only the custom classification head is trained (Dropout → Linear → LayerNorm → GELU → Dropout → Linear)
- Learning rate: 5e-4, with cosine annealing
- Purpose: adapt quickly to the new task without damaging the pretrained features

*Phase 2 (remaining epochs)*

- The whole network is unfrozen and fine-tuned together
- Backbone learning rate: 2e-5 (10× slower than the head)
- Classification head learning rate: 2e-4
- Purpose: fine-tune the whole model

The first 5 epochs also use a linear learning-rate warmup to avoid unstable early training.

**Changing training settings**

Open `train.py` and edit the `CFG` dictionary at the top:

```python
"batch_size"          : 32    # reduce to 16 or 8 if you run out of GPU memory
"num_epochs"          : 100   # total number of epochs
"early_stop_patience" : 15    # stop if there is no improvement for this many epochs
"head_only_epochs"    : 10    # number of Phase 1 (frozen backbone) epochs
"data_dir"            : "./skin_cancer_data"  # dataset path
"img_size"            : 224   # input image size (not recommended to change)
```

**Resuming training (automatic)**

If training is interrupted, running `python train.py` again resumes automatically from the last checkpoint. No extra steps are needed. Checkpoint data is saved in `checkpoint_latest.pth`.

**Training from scratch**

In Windows CMD:

```bash
del checkpoint_latest.pth
python train.py
```

**Monitoring training (TensorBoard)**

In a second CMD window, run:

```bash
tensorboard --logdir ./runs/skin_cancer_v2
```

Then open http://localhost:6006 in your browser. You can view loss, accuracy, macro F1, per-class F1, learning rate and AUC curves.

**Files generated after training**

```
best_model.pth              ← best model weights (highest validation macro F1)
best_model.onnx             ← ONNX format (for production deployment)
best_model_scripted.pt      ← TorchScript format (loadable directly in C++)
checkpoint_latest.pth       ← checkpoint for resuming training
training_log.csv            ← metrics for each epoch (opens in Excel)
reports/
  confusion_matrix_best.png       ← confusion matrix
  roc_curves_best.png             ← ROC curves (with AUC)
  classification_report_best.txt  ← detailed classification report
```

**How to tell whether training is going well**

Good signs:

- Validation macro F1 rises steadily and reaches a new best within every 15 epochs
- Training and validation loss fall together, with a gap of no more than 0.2
- AUC above 0.80 (late in Phase 2)

Warning signs:

- Validation loss rises while training loss keeps falling → overfitting; try fewer epochs
- F1 is very low for all classes (< 0.3) and does not improve → check the dataset
- Melanoma F1 stays below 0.4 → severe class imbalance; add more melanoma images

---

## 7. Command-line inference (`infer.py`)

`infer.py` is a command-line inference script, suitable for automation. If you only need the visual interface, use `app.py` instead.

**Single image (TTA on by default)**

```bash
python infer.py --image path/to/image.jpg
```

**Batch inference on a folder**

```bash
python infer.py --folder ./test_images
```

**Save results as JSON**

```bash
python infer.py --folder ./test_images --output results.json
```

**Turn off TTA (about 8× faster, slightly less accurate)**

```bash
python infer.py --image image.jpg --no-tta
```

**Load a TorchScript model**

```bash
python infer.py --image image.jpg --model best_model_scripted.pt --torchscript
```

**About TTA (test-time augmentation)**

TTA creates 8 transformed versions of each image:

1. Original
2. Horizontal flip
3. Vertical flip
4. Horizontal + vertical flip
5. 90° rotation
6. 180° rotation
7. 270° rotation
8. 1.15× zoom with centre crop

The probabilities from the 8 predictions are averaged, which usually improves accuracy by 0.5–2%. Results also include an uncertainty estimate (± standard deviation); a smaller value means the model is more consistent.

---

## 8. Web interface (`app.py`)

`app.py` is the recommended way to use the system. It provides a complete graphical interface. The interface is in Chinese, so the on-screen labels are given in brackets below.

**Getting started**

1. Copy the skin images you want to analyse into the `orgimg/` folder in the project folder (it is created automatically on first run).
2. Start the server:
   ```bash
   python app.py
   ```
3. Open http://localhost:5000 in your browser (Chrome or Edge recommended).
4. Click **Start batch analysis** (⚡ 开始批量分析) and wait for processing to finish.
5. Results appear on the right-hand side of the page as they are ready.

**Interface**

*Left panel: images to process (📁 待处理图像)*

- Lists all images in `orgimg/`
- ✅ Green = processed
- ⚙️ Flashing blue = processing
- A progress bar (处理进度) shows overall progress, and the image currently being processed (正在处理)

*Results area (📊 分析结果), one card per image*

- Left column: original image (原始图像)
- Middle column: Grad-CAM heatmap (异常热力图 (Grad-CAM))
  - 🔴 Red areas (红色区域) = activation related to melanoma (黑色素瘤相关异常激活)
  - 🔵 Blue areas (蓝色区域) = activation related to other skin cancers (皮肤癌相关异常激活)
  - 🟢 Green areas (绿色区域) = features related to non-cancerous / healthy skin (非癌症/健康相关特征)
  - Darker colour = greater influence on that class
- Right column: probability bars for the three classes (各类别概率), confidence of the top prediction (主预测置信度), risk level and a link to the report (📄 查看风险报告)

*Summary badges above the results*

- Melanoma high risk (黑色素瘤高危)
- Other skin cancer high risk (皮肤癌高危)
- Biopsy recommended (建议活检)
- Low risk (低风险)

*Top status bar*

- Shows the GPU model in use (or CPU)
- Shows whether the model loaded successfully; if not, it shows "Model not loaded" (⚠️ 模型未加载) or "Cannot connect" (❌ 无法连接)

**Risk levels**

| Risk level | On-screen label | Rule | Advice |
| --- | --- | --- | --- |
| High (red) | 高风险 | Melanoma probability > 50% | Seek medical attention promptly |
| High (yellow) | 高风险 | Other skin cancer probability > 50% | Seek medical attention promptly |
| Biopsy recommended (blue) | 建议切片活检 | Melanoma or other skin cancer probability > 30% | Consider a biopsy |
| Low (green) | 低风险 | Both below 30%, or the top prediction is non-cancerous | Lower risk for now; keep monitoring |

**Stopping the server**

Press `Ctrl+C` in the CMD window running `python app.py`.

---

## 9. Output files

Each processed image gets its own subfolder in `processedimg/`:

```
processedimg/
  ├── imageA/
  │   ├── original.jpg      ← copy of the original image
  │   ├── heatmap.jpg       ← three-colour Grad-CAM heatmap overlay
  │   └── risk_report.txt   ← detailed text risk report
  │
  ├── imageB/
  │   ├── original.jpg
  │   ├── heatmap.jpg
  │   └── risk_report.txt
  └── ...
```

**Contents of `risk_report.txt`** (written in Chinese)

- File name and processing time
- Top predicted class and confidence
- Risk level and medical advice
- Text-based probability bars for the three classes
- Explanation of the heatmap colours
- Risk level definitions
- Disclaimer

---

## 10. Troubleshooting

**CUDA out of memory**

Open `train.py` and change `batch_size` in `CFG` from 32 to 16 or 8.

**`ModuleNotFoundError: No module named 'xxx'`**

Reinstall the missing dependency as described in section 4:

```bash
pip install xxx
```

**The browser says "This site can't be reached" after starting `app.py`**

Check that there are no errors in the CMD window and that port 5000 is free. To use a different port, edit the last line of `app.py`:

```python
app.run(host="0.0.0.0", port=5001, ...)
```

Then open http://localhost:5001.

**The interface shows "Model not loaded" (⚠️ 模型未加载, yellow warning at the top)**

Make sure `best_model.pth` is in the same folder as `app.py`. If you have not trained a model yet, run `python train.py` first.

**AUC stays at 0.5000 during training**

This bug has been fixed in the current `train.py`. If it still happens, check that you are using the latest version.

**`app.py` reports "cv2 not found" when processing images**

Reinstall a compatible version of OpenCV:

```bash
pip install "numpy==1.26.4"
pip install "opencv-python-headless==4.9.0.80"
```

**The `rm` command doesn't work (Windows)**

On Windows, use `del` instead of `rm`, and `rmdir` instead of `rm -rf`:

```bash
del checkpoint_latest.pth
rmdir /s /q reports runs
```

**Macro F1 doesn't improve after several epochs**

Possible causes and fixes:

- Dataset too small → increase each class to 500+ images
- Learning rate too high → change `head_lr` from 5e-4 to 1e-4
- Severe class imbalance → add more images of the minority class (melanoma)

**The heatmap is completely dark or barely visible**

Grad-CAM activation for that class is very weak. This usually means the model is very confident about the image (probability close to 100%), so the gradients are close to zero and the heatmap is faint. This is normal.

---

## 11. Important notes

**⚠️ Medical disclaimer**

All outputs of this system (classifications, risk levels and heatmaps) are generated automatically by an AI model and are for research and reference only. This system cannot replace a diagnosis by a qualified dermatologist. If you have any concerns about a skin lesion, see a doctor promptly.

**⚠️ Data privacy**

The system runs entirely on your own computer. No images or data are uploaded to any server. The web service started by `app.py` is only accessible from your own machine.

**⚠️ Model limitations**

- The model was trained on a specific dataset and may perform poorly on images unlike its training data
- Image quality (lighting, sharpness, angle) strongly affects predictions
- Probabilities reflect the model's confidence, not clinical diagnostic accuracy
- Use standard dermoscopy images; accuracy on ordinary phone photos is lower
- Public dermoscopy datasets mostly contain lighter skin tones, so performance on darker skin may be worse

**⚠️ Version compatibility**

- Trained model weights are tied to the network architecture in `train.py`
- If you change `build_model()` (the network structure), delete the old `best_model.pth` and retrain from scratch
- The model architecture in `app.py` and `infer.py` must exactly match `train.py`

---

## Quick start (three steps)

1. **Prepare the data**
   Put the images into `skin_cancer_data/train/` and `skin_cancer_data/val/`, sorted by class.
2. **Train the model**
   ```bash
   python train.py
   ```
   `best_model.pth` is generated automatically when training finishes.
3. **Start the web interface**
   Put the images to analyse in `orgimg/`, then run:
   ```bash
   python app.py
   ```
   Open http://localhost:5000 and click **Start batch analysis** (⚡ 开始批量分析).
