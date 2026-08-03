"""
app.py — 皮肤癌 AI 风险评估系统 (定制高阶判定逻辑版)
=====================================================
安装依赖:
  pip install flask opencv-python-headless

运行方式:
  python app.py
  浏览器打开: http://localhost:5000

目录结构 (与 train.py 放在同一文件夹):
  orgimg/          ← 将待检测图像放入此文件夹
  processedimg/    ← 自动生成，每张图对应一个子文件夹
  best_model.pth   ← train.py 训练产生的最优模型
=====================================================
"""

import os
import sys
import json
import time
import shutil
import threading
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small
from flask import Flask, send_file, jsonify, request

# ══════════════════════════════════════════════════════════════
# 1. 配置 (CLASS_NAMES 须与 train.py 目录排序一致)
# ══════════════════════════════════════════════════════════════
MODEL_PATH    = "best_model.pth"
ORG_DIR       = Path("orgimg")
OUT_DIR       = Path("processedimg")
IMG_SIZE      = 224
NUM_CLASSES   = 3
CLASS_NAMES   = ["0_melanoma", "1_other_cancer", "2_other_conditions"]
CLASS_LABELS  = ["黑色素瘤", "皮肤癌", "非癌症/健康"]
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ─── 全局状态 ─────────────────────────────────────────────────
_model = None
_state = {
    "running": False, "total": 0, "done": 0,
    "current": "", "results": [], "error": None,
}
_lock = threading.Lock()

# ══════════════════════════════════════════════════════════════
# 2. 模型架构 (与 train.py build_model 完全一致)
# ══════════════════════════════════════════════════════════════
def _build_arch() -> nn.Module:
    m = convnext_small(weights=None)
    in_f = m.classifier[2].in_features   # 768
    m.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_f, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, NUM_CLASSES),
    )
    return m


def load_model() -> bool:
    global _model
    if not Path(MODEL_PATH).exists():
        return False
    m = _build_arch()
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    m.load_state_dict(state)
    _model = m.to(device).eval()
    return True

# ══════════════════════════════════════════════════════════════
# 3. Grad-CAM (ConvNeXt-S 适配)
# ══════════════════════════════════════════════════════════════
class GradCAM:
    """
    标准 Grad-CAM，hook 挂在 model.features[7]
    (ConvNeXt-S 最后 Stage，输出 B×768×7×7)。
    三种疾病类别分别独立计算，叠加成复合热力图:
      Red  = Melanoma      (黑色素瘤)
      Blue = Other Cancer  (皮肤癌)
      Green= Non-cancer/Healthy (非癌症/健康)
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self._acts = self._grads = None
        self._fh = model.features[7].register_forward_hook(
            lambda m, i, o: setattr(self, "_acts", o.detach())
        )
        self._bh = model.features[7].register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_grads", go[0].detach())
        )

    def compute(self, inp: torch.Tensor, cls_idx: int) -> np.ndarray:
        """返回 [0,1] 归一化 CAM，shape (H, W)。"""
        self.model.zero_grad()
        with torch.set_grad_enabled(True):
            out = self.model(inp)
            out[0, cls_idx].backward()

        w   = self._grads.mean(dim=[2, 3], keepdim=True)   # (1, 768, 1, 1)
        cam = F.relu((w * self._acts).sum(dim=1))           # (1, H, W)
        cam = cam.squeeze(0).cpu().numpy()

        mn, mx = cam.min(), cam.max()
        return (cam - mn) / (mx - mn + 1e-8)

    def remove(self):
        self._fh.remove()
        self._bh.remove()

# ══════════════════════════════════════════════════════════════
# 4. 图像处理与风险评测决策矩阵
# ══════════════════════════════════════════════════════════════
_tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def eval_custom_risk(probs: list, pred_i: int) -> tuple:
    """
    自适应临床判定规则：
    - 黑色素瘤一栏(idx 0) > 50% ➡️ HIGH_MEL (高风险，红色)
    - 皮肤癌一栏(idx 1) > 50% ➡️ HIGH_CAN (高风险，黄色)
    - 两者中的任意一者 > 30% ➡️ BIOPSY (建议切片活检，蓝色)
    - 最后一栏非癌症/健康(idx 2)无论多少 ➡️ LOW (低风险，绿色)
    - 其余情况(均低于30%) ➡️ LOW (低风险，绿色)
    """
    p_mel = float(probs[0])
    p_can = float(probs[1])
    
    # 最后一栏非癌症/健康(idx 2)无论多少，或者预测出来就是健康类
    if pred_i == 2:
        return "LOW", "低风险", "#10b981"
        
    # 黑色素瘤超过百分之50标红写高风险
    if p_mel > 0.50:
        return "HIGH_MEL", "高风险", "#ef4444"
        
    # 皮肤癌一栏超过百分之50标黄写高风险
    if p_can > 0.50:
        return "HIGH_CAN", "高风险", "#f59e0b"
        
    # 两者中的任意一者高于百分之30则表蓝显示建议切片活检
    if p_mel > 0.30 or p_can > 0.30:
        return "BIOPSY", "建议切片活检", "#3b82f6"
        
    # 都低于百分之30则显示绿色低风险
    return "LOW", "低风险", "#10b981"


def composite_heatmap(img_rgb: np.ndarray,
                      cams: list, probs: list) -> np.ndarray:
    """
    将三个类别的 Grad-CAM 融合为 RGB 复合热力图后叠加在原图上。
    各通道强度 = CAM × 该类别预测概率。
    """
    H, W  = img_rgb.shape[:2]
    heat  = np.zeros((H, W, 3), dtype=np.float32)   # RGB float

    for i, (cam, p) in enumerate(zip(cams, probs)):
        c = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR) * float(p)
        if   i == 0: heat[:, :, 0] += c  # Melanoma  → Red
        elif i == 1: heat[:, :, 2] += c  # Cancer    → Blue
        else:        heat[:, :, 1] += c  # Other     → Green

    mx = heat.max()
    if mx > 0:
        heat /= mx

    # Alpha-blend
    alpha = heat.max(axis=2, keepdims=True).clip(0, 1) * 0.7
    orig  = img_rgb.astype(np.float32) / 255.0
    blend = orig * (1.0 - alpha) + heat * alpha
    return (blend.clip(0, 1) * 255).astype(np.uint8)


def process_one(gc: GradCAM, img_path: Path, sub_dir: Path) -> dict:
    """处理单张图像并生成定制化学术风险报告。"""
    img_pil = Image.open(img_path).convert("RGB")
    img_np  = np.array(img_pil)

    inp = _tfm(img_pil).unsqueeze(0).to(device)

    # ── 推理 ──────────────────────────────────────────────
    with torch.no_grad():
        logits = _model(inp)
        probs  = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        probs  = probs / probs.sum()   # 保证精确归一

    # ── Grad-CAM ──────────────────────────────────────────
    cams = [gc.compute(inp, i) for i in range(NUM_CLASSES)]

    pred_i = int(np.argmax(probs))
    pred_p = float(probs[pred_i])
    
    # ⚠️ 加载定制临床策略评估规则
    r_en, r_cn, r_color = eval_custom_risk(probs.tolist(), pred_i)

    # ── 合成热力图 ─────────────────────────────────────────
    overlay = composite_heatmap(img_np, cams, probs.tolist())

    # ── 保存文件 ───────────────────────────────────────────
    sub_dir.mkdir(parents=True, exist_ok=True)
    suf      = img_path.suffix.lower()
    orig_out = sub_dir / f"original{suf}"
    heat_out = sub_dir / f"heatmap{suf}"
    rep_out  = sub_dir / "risk_report.txt"

    shutil.copy2(img_path, orig_out)
    cv2.imwrite(str(heat_out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # ── TXT 风险报告 ───────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    advice = {
        "HIGH_MEL": "强烈建议立即前往皮肤科门诊进行病理学活检排查",
        "HIGH_CAN": "强烈建议尽快预约皮肤专科医生诊治并保持随访",
        "BIOPSY"  : "病变特征激活较强，建议采取局部切片活检以明确诊断",
        "LOW"     : "暂时风险较低，建议继续保持常规皮肤监测与自我健康管理",
    }[r_en]

    lines = [
        "═" * 54,
        "           皮肤病变 AI 风险评估报告 (定制判定版)",
        "═" * 54,
        f"  文件名    : {img_path.name}",
        f"  处理时间  : {ts}",
        f"  运行设备  : {str(device).upper()}",
        f"  模型      : ConvNeXt-Small (EMA 权重)",
        "",
        "─" * 54,
        "  ◆ 主要预测结果",
        "─" * 54,
        f"    预测类别 : {CLASS_LABELS[pred_i]}  ({CLASS_NAMES[pred_i]})",
        f"    置信度   : {pred_p:.2%}",
        f"    风险等级 : {r_cn}",
        f"    专业建议 : {advice}",
        "",
        "─" * 54,
        "  ◆ 各类别概率分布",
        "─" * 54,
    ]
    for i in range(NUM_CLASSES):
        p   = float(probs[i])
        bar = "█" * int(p * 26) + "░" * (26 - int(p * 26))
        # 针对当前类单独用自变量函数计算评估状态
        _, temp_cn, _ = eval_custom_risk([p if k==i else 0 for k in range(3)], i)
        lines.append(f"    {CLASS_LABELS[i]:<6}  |{bar}|  {p:.2%}  [{temp_cn}]")

    lines += [
        "",
        "─" * 54,
        "  ◆ 热力图颜色说明",
        "─" * 54,
        "    🔴 红色区域  →  黑色素瘤 (Melanoma) 相关异常",
        "    🔵 蓝色区域  →  皮肤癌 (Other Cancer) 相关异常",
        "    🟢 绿色区域  →  非癌症/健康 相关特征激活",
        "    颜色深度代表该区域对对应类别的激活强度",
        "",
        "─" * 54,
        "  ◆ 决策评级规则说明",
        "─" * 54,
        f"    高风险 (红色) : 黑色素瘤置信度 > 50%",
        f"    高风险 (黄色) : 其他皮肤癌置信度 > 50%",
        f"    建议活检 (蓝色) : 黑色素瘤或皮肤癌任意概率 > 30%",
        f"    低风险 (绿色) : 均低于 30% 或预测为非癌症健康状态",
        "",
        "═" * 54,
        "  ⚠️  免责声明",
        "═" * 54,
        "  此报告由 AI 模型自动生成，仅供参考，",
        "  不能替代持牌皮肤科医生的专业诊断。",
        "  如有任何疑虑，请立即就医。",
        "═" * 54,
    ]
    rep_out.write_text("\n".join(lines), encoding="utf-8")

    def fwd(p): return str(p).replace("\\", "/")   # Windows 路径修复

    return {
        "filename"   : img_path.name,
        "stem"       : img_path.stem,
        "pred_i"     : pred_i,
        "pred_label" : CLASS_LABELS[pred_i],
        "pred_name"  : CLASS_NAMES[pred_i],
        "pred_prob"  : pred_p,
        "risk_en"    : r_en,
        "risk_cn"    : r_cn,
        "risk_color" : r_color,
        "probs"      : {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "orig"       : fwd(orig_out),
        "heat"       : fwd(heat_out),
        "report"     : fwd(rep_out),
    }


def run_batch():
    """后台线程：批量处理 orgimg/ 中所有图像。"""
    global _state
    images = sorted([
        p for p in ORG_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ])
    with _lock:
        _state.update({
            "running": True, "total": len(images),
            "done": 0, "current": "", "results": [], "error": None,
        })

    OUT_DIR.mkdir(exist_ok=True)
    gc = GradCAM(_model)

    try:
        for img_path in images:
            with _lock:
                _state["current"] = img_path.name
            try:
                r = process_one(gc, img_path, OUT_DIR / img_path.stem)
                with _lock:
                    _state["results"].append(r)
            except Exception as e:
                with _lock:
                    _state["results"].append({"filename": img_path.name, "error": str(e)})
            with _lock:
                _state["done"] += 1
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
    finally:
        gc.remove()
        with _lock:
            _state["running"] = False
            _state["current"] = ""

# ══════════════════════════════════════════════════════════════
# 5. Flask 路由
# ══════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder=None)


@app.route("/")
def index():
    return HTML_PAGE, 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/api/status")
def api_status():
    return jsonify({
        "model_loaded"  : _model is not None,
        "model_path"    : MODEL_PATH,
        "device"        : str(device),
        "cuda"          : torch.cuda.is_available(),
        "orgimg_exists" : ORG_DIR.exists(),
    })


@app.route("/api/scan")
def api_scan():
    ORG_DIR.mkdir(exist_ok=True)
    files = sorted([
        p.name for p in ORG_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXT
    ])
    return jsonify({"files": files, "count": len(files)})


@app.route("/api/process", methods=["POST"])
def api_process():
    if _model is None:
        return jsonify({"error": f"模型未加载，请确保 '{MODEL_PATH}' 存在"}), 400
    with _lock:
        if _state["running"]:
            return jsonify({"error": "正在处理中，请稍候..."}), 400
    threading.Thread(target=run_batch, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/progress")
def api_progress():
    with _lock:
        return jsonify(dict(_state))


@app.route("/img/<path:fpath>")
def serve_img(fpath):
    abs_path = Path(".").resolve() / fpath.replace("\\", "/")
    if not abs_path.exists():
        return "Not found", 404
    return send_file(str(abs_path))


# ══════════════════════════════════════════════════════════════
# 6. 前端 HTML (重写前端指标分类渲染系统)
# ══════════════════════════════════════════════════════════════
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>皮肤癌 AI 风险评估系统</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --bg:#060c1a;--card:#0d1627;--card2:#111e35;--border:#1c3355;
  --accent:#2563eb;--text:#e2e8f0;--dim:#6b8aad;
  --high_mel:#ef4444;--high_can:#f59e0b;--biopsy:#3b82f6;--low:#10b981;
  --mel:#ef4444;--can:#3b82f6;--oth:#22c55e;
}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;height:100vh;overflow:hidden;display:flex;flex-direction:column}

/* Grid background */
body::before{content:'';position:fixed;inset:0;
  background-image:linear-gradient(rgba(37,99,235,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(37,99,235,.04) 1px,transparent 1px);
  background-size:48px 48px;pointer-events:none;z-index:0}

/* Header */
header{position:relative;z-index:10;background:linear-gradient(135deg,#080f20,#0d1c38);
  border-bottom:1px solid var(--border);padding:14px 28px;
  display:flex;align-items:center;gap:16px;flex-shrink:0}
.logo{width:42px;height:42px;border-radius:12px;
  background:linear-gradient(135deg,#1d4ed8,#7c3aed);
  display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0}
header h1{font-size:18px;font-weight:700;letter-spacing:.01em}
header p{font-size:11px;color:var(--dim);margin-top:2px}
#status-pill{margin-left:auto;padding:5px 14px;border-radius:20px;font-size:12px;font-weight:700;border:1px solid;white-space:nowrap}
.pill-ok{color:var(--low);border-color:var(--low);background:rgba(16,185,129,.1)}
.pill-warn{color:var(--high_can);border-color:var(--high_can);background:rgba(245,158,11,.1)}
.pill-err{color:var(--high_mel);border-color:var(--high_mel);background:rgba(239,68,68,.1)}

/* Layout */
.body-wrap{display:flex;flex:1;overflow:hidden;position:relative;z-index:1}

/* Sidebar */
.sidebar{width:300px;flex-shrink:0;background:var(--card);border-right:1px solid var(--border);display:flex;flex-direction:column;overflow:hidden}
.sb-top{padding:18px;border-bottom:1px solid var(--border)}
.sb-top h2{font-size:11px;font-weight:700;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px}
.sb-meta{font-size:12px;color:var(--dim);margin-bottom:12px}
.sb-meta b{color:var(--text)}

/* Buttons */
.btn{display:flex;align-items:center;justify-content:center;gap:8px;
  width:100%;padding:10px;border-radius:8px;border:none;
  font-size:13px;font-weight:700;cursor:pointer;transition:.2s}
.btn-primary{background:linear-gradient(135deg,#1d4ed8,#2563eb);color:#fff;
  box-shadow:0 4px 14px rgba(37,99,235,.35)}
.btn-primary:hover:not(:disabled){transform:translateY(-1px);box-shadow:0 6px 20px rgba(37,99,235,.45)}
.btn-primary:disabled{opacity:.45;cursor:not-allowed;transform:none}

/* Progress */
.prog-wrap{padding:14px 18px;border-bottom:1px solid var(--border);display:none}
.prog-wrap.on{display:block}
.prog-meta{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);margin-bottom:7px}
.prog-meta span{color:var(--text);font-weight:600}
.prog-bar{height:5px;background:rgba(255,255,255,.07);border-radius:3px;overflow:hidden;margin-bottom:7px}
.prog-fill{height:100%;background:linear-gradient(90deg,#1d4ed8,#7c3aed);border-radius:3px;transition:width .35s ease;width:0%}
.prog-cur{font-size:11px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* File list */
.file-list{flex:1;overflow-y:auto;padding:10px}
.fi{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:7px;font-size:12px;color:var(--dim);transition:.15s;cursor:default}
.fi:hover{background:rgba(255,255,255,.03)}
.fi-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fi.done{color:var(--low)}
.fi.going{color:var(--accent)}
.fi.fail{color:var(--high_mel)}
.fi-empty{padding:20px;text-align:center;font-size:12px;color:var(--dim)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}
.fi.going{animation:blink 1s infinite}

/* Results panel */
.rp{flex:1;overflow-y:auto;padding:22px}
.rp-header{display:flex;align-items:center;gap:14px;margin-bottom:18px}
.rp-header h2{font-size:17px;font-weight:700}
.stats{display:flex;gap:10px;flex-wrap:wrap}
.sbadge{padding:4px 12px;border-radius:14px;font-size:11px;font-weight:700;border:1px solid}

/* 学术分类卡片配色 (CSS 变量解耦) */
.s_high_mel{color:var(--high_mel);border-color:rgba(239,68,68,.4);background:rgba(239,68,68,.1)}
.s_high_can{color:var(--high_can);border-color:rgba(245,158,11,.4);background:rgba(245,158,11,.1)}
.s_biopsy{color:var(--biopsy);border-color:rgba(59,130,246,.4);background:rgba(59,130,246,.1)}
.s_low{color:var(--low);border-color:rgba(16,185,129,.4);background:rgba(16,185,129,.1)}

/* Empty state */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
  min-height:400px;color:var(--dim);text-align:center;gap:14px}
.empty-ico{font-size:60px;opacity:.25}
.empty h3{font-size:17px;color:var(--text)}
.empty p{font-size:13px;max-width:260px;line-height:1.65}
.empty code{background:rgba(255,255,255,.07);padding:2px 7px;border-radius:5px;font-size:12px}

/* Result cards */
.rg{display:flex;flex-direction:column;gap:18px}

.rcard{background:var(--card);border:1px solid var(--border);border-radius:16px;overflow:hidden;
  transition:border-color .2s,transform .2s}
.rcard:hover{border-color:rgba(37,99,235,.45);transform:translateY(-2px)}

.rcard-head{padding:14px 18px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--border)}
.rcard-head .fname{flex:1;font-size:14px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* 定制化 Badge 样式 */
.rbadge{padding:4px 13px;border-radius:14px;font-size:11px;font-weight:800;border:1px solid;white-space:nowrap}
.rbHIGH_MEL{color:var(--high_mel);border-color:var(--high_mel);background:rgba(239,68,68,.12)}
.rbHIGH_CAN{color:var(--high_can);border-color:var(--high_can);background:rgba(245,158,11,.12)}
.rbBIOPSY{color:var(--biopsy);border-color:var(--biopsy);background:rgba(59,130,246,.12)}
.rbLOW{color:var(--low);border-color:var(--low);background:rgba(16,185,129,.12)}

.rcard-body{display:grid;grid-template-columns:1fr 1fr 260px}

.img-col{padding:14px;border-right:1px solid var(--border)}
.img-col label{display:block;font-size:10px;font-weight:700;color:var(--dim);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px}
.img-col img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:10px;
  border:1px solid var(--border);display:block}

.legend{margin-top:10px;display:flex;flex-direction:column;gap:4px}
.leg-row{display:flex;align-items:center;gap:6px;font-size:10px;color:var(--dim)}
.leg-dot{width:9px;height:9px;border-radius:50%;flex-shrink:0}

.prob-col{padding:16px 18px}
.prob-col h4{font-size:10px;font-weight:700;color:var(--dim);
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:14px}
.prow{margin-bottom:12px}
.prow-top{display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px}
.prow-top .cls{font-weight:700}
.prow-top .val{color:var(--dim)}
.pbar-bg{height:7px;background:rgba(255,255,255,.06);border-radius:4px;overflow:hidden}
.pbar-f{height:100%;border-radius:4px;transition:width .9s cubic-bezier(.34,1.56,.64,1)}
.bm{background:linear-gradient(90deg,#991b1b,#ef4444)}
.bc{background:linear-gradient(90deg,#eab308,#f59e0b)}
.bo{background:linear-gradient(90deg,#166534,#22c55e)}

.pred-box{margin-top:14px;background:var(--card2);border-radius:10px;padding:12px 14px}
.pred-box .pcls{font-size:14px;font-weight:700;margin-bottom:2px}
.pred-box .pconf{font-size:12px;color:var(--dim)}

.download-link{display:inline-flex;align-items:center;gap:5px;
  margin-top:10px;font-size:11px;color:var(--accent);text-decoration:none;
  border:1px solid rgba(37,99,235,.3);padding:4px 10px;border-radius:6px;
  transition:.15s}
.download-link:hover{background:rgba(37,99,235,.1)}

.err-body{padding:18px;color:var(--high_mel);font-size:12px}

/* Scrollbar */
::-webkit-scrollbar{width:4px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

/* Toast */
.toast{position:fixed;bottom:22px;right:22px;z-index:1000;
  background:var(--card2);border:1px solid var(--border);
  border-radius:10px;padding:13px 16px;font-size:13px;
  max-width:340px;transform:translateY(70px);opacity:0;
  transition:.3s ease}
.toast.show{transform:translateY(0);opacity:1}
.toast.ok{border-left:3px solid var(--low)}
.toast.err{border-left:3px solid var(--high_mel)}
</style>
</head>
<body>

<header>
  <div class="logo">🔬</div>
  <div>
    <h1>皮肤癌 AI 风险评估系统</h1>
    <p>ConvNeXt-S &nbsp;·&nbsp; Grad-CAM 多类热力图 &nbsp;·&nbsp; EMA 权重推理</p>
  </div>
  <div id="status-pill" class="pill-warn">加载中…</div>
</header>

<div class="body-wrap">

  <!-- ── Sidebar ── -->
  <div class="sidebar">
    <div class="sb-top">
      <h2>📁 待处理图像</h2>
      <div class="sb-meta">来自 <b>orgimg/</b> &nbsp;共 <b id="fcnt">0</b> 张</div>
      <button class="btn btn-primary" id="proc-btn" onclick="startProc()">
        ⚡ 开始批量分析
      </button>
    </div>

    <div class="prog-wrap" id="prog-wrap">
      <div class="prog-meta">
        <span>处理进度</span>
        <span id="prog-txt">0 / 0</span>
      </div>
      <div class="prog-bar"><div class="prog-fill" id="prog-fill"></div></div>
      <div class="prog-cur" id="prog-cur">准备中…</div>
    </div>

    <div class="file-list" id="file-list">
      <div class="fi-empty">扫描中…</div>
    </div>
  </div>

  <!-- ── Results ── -->
  <div class="rp">
    <div class="rp-header">
      <h2>📊 分析结果</h2>
      <div class="stats" id="stats" style="display:none">
        <span class="sbadge s_high_mel" id="s_high_mel">黑色素瘤高危 0</span>
        <span class="sbadge s_high_can" id="s_high_can">皮肤癌高危 0</span>
        <span class="sbadge s_biopsy" id="s_biopsy">建议活检 0</span>
        <span class="sbadge s_low" id="s_low">低风险 0</span>
      </div>
    </div>

    <div class="empty" id="empty">
      <div class="empty-ico">🧬</div>
      <h3>等待分析</h3>
      <p>将皮肤图像放入 <code>orgimg/</code> 文件夹<br>然后点击「开始批量分析」</p>
    </div>

    <div class="rg" id="rg" style="display:none"></div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
"use strict";
let files=[], seen=new Set(), poll=null, cnt={HIGH_MEL:0,HIGH_CAN:0,BIOPSY:0,LOW:0};

const clsInfo = {
  "0_melanoma"       : {label:"黑色素瘤", bar:"bm", color:"var(--mel)"},
  "1_other_cancer"   : {label:"皮肤癌",   bar:"bc", color:"var(--high_can)"},
  "2_other_conditions":{label:"非癌症/健康", bar:"bo", color:"var(--oth)"},
};

// ── Init ─────────────────────────────────────────────────────
window.addEventListener("load", async ()=>{
  await checkStatus();
  await scanFiles();
});

async function checkStatus(){
  try{
    const d = await get("/api/status");
    const p = document.getElementById("status-pill");
    if(d.model_loaded){
      p.textContent = "✅ "+d.device.toUpperCase();
      p.className   = "pill-ok";
    } else {
      p.textContent = "⚠️ 模型未加载";
      p.className   = "pill-warn";
    }
  }catch(e){
    document.getElementById("status-pill").textContent="❌ 无法连接";
    document.getElementById("status-pill").className="pill-err";
  }
}

async function scanFiles(){
  const d = await get("/api/scan");
  files = d.files;
  document.getElementById("fcnt").textContent = d.count;
  renderFileList(new Set(), "");
}

function renderFileList(doneSet, cur){
  const el = document.getElementById("file-list");
  if(!files.length){
    el.innerHTML = '<div class="fi-empty">orgimg/ 中暂无图像</div>';
    return;
  }
  el.innerHTML = files.map(f=>{
    let cls="", ico="🖼️";
    if(doneSet.has(f)){cls="done"; ico="✅";}
    else if(f===cur){cls="going"; ico="⚙️";}
    return `<div class="fi ${cls}"><span>${ico}</span><span class="fi-name">${f}</span></div>`;
  }).join("");
}

// ── Process ───────────────────────────────────────────────────
async function startProc(){
  const btn = document.getElementById("proc-btn");
  btn.disabled = true; btn.textContent="⏳ 处理中…";
  cnt = {HIGH_MEL:0,HIGH_CAN:0,BIOPSY:0,LOW:0};
  seen = new Set();
  document.getElementById("rg").innerHTML = "";
  document.getElementById("rg").style.display = "none";
  document.getElementById("empty").style.display = "none";
  document.getElementById("stats").style.display = "none";

  try{
    const r = await post("/api/process");
    if(r.error){ toast(r.error,"err"); btn.disabled=false; btn.textContent="⚡ 开始批量分析"; return; }
    document.getElementById("prog-wrap").classList.add("on");
    poll = setInterval(doPoll, 600);
  }catch(e){
    toast("启动失败: "+e.message,"err");
    btn.disabled=false; btn.textContent="⚡ 开始批量分析";
  }
}

async function doPoll(){
  const s = await get("/api/progress");
  const pct = s.total>0 ? s.done/s.total*100 : 0;
  document.getElementById("prog-fill").style.width = pct+"%";
  document.getElementById("prog-txt").textContent  = s.done+" / "+s.total;
  document.getElementById("prog-cur").textContent  =
    s.current ? "正在处理: "+s.current : (s.running?"准备中…":"✅ 完成");

  const doneNames = new Set(s.results.map(r=>r.filename));
  renderFileList(doneNames, s.current);

  const grid = document.getElementById("rg");
  grid.style.display = "flex";
  document.getElementById("empty").style.display = "none";

  s.results.forEach(r=>{
    if(!seen.has(r.filename)){
      seen.add(r.filename);
      grid.insertAdjacentHTML("beforeend", r.error ? errCard(r) : resultCard(r));
      if(!r.error){ cnt[r.risk_en]=(cnt[r.risk_en]||0)+1; updateStats(); }
    }
  });

  if(!s.running){
    clearInterval(poll);
    const btn = document.getElementById("proc-btn");
    btn.disabled=false; btn.textContent="⚡ 重新分析";
    if(s.error) toast("处理错误: "+s.error,"err");
    else toast(`✅ 完成！${s.done} 张已保存至 processedimg/`,"ok");
  }
}

function updateStats(){
  document.getElementById("stats").style.display="flex";
  document.getElementById("s_high_mel").textContent="黑色素瘤高危 "+cnt.HIGH_MEL;
  document.getElementById("s_high_can").textContent="皮肤癌高危 "+cnt.HIGH_CAN;
  document.getElementById("s_biopsy").textContent="建议活检 "+cnt.BIOPSY;
  document.getElementById("s_low").textContent="低风险 "+cnt.LOW;
}

// ── Cards ─────────────────────────────────────────────────────
function resultCard(r){
  const bars = Object.entries(r.probs).map(([name, p])=>{
    const info = clsInfo[name]||{label:name,bar:"bm",color:"#888"};
    const pct  = (p*100).toFixed(1);
    return `<div class="prow">
      <div class="prow-top">
        <span class="cls" style="color:${info.color}">${info.label}</span>
        <span class="val">${pct}%</span>
      </div>
      <div class="pbar-bg"><div class="pbar-f ${info.bar}" style="width:${pct}%"></div></div>
    </div>`;
  }).join("");

  const repUrl = "/img/"+r.report;

  return `<div class="rcard">
  <div class="rcard-head">
    <span style="font-size:18px">🖼️</span>
    <span class="fname">${r.filename}</span>
    <span class="rbadge rb${r.risk_en}">${r.risk_cn}</span>
  </div>
  <div class="rcard-body">
    <div class="img-col">
      <label>原始图像</label>
      <img src="/img/${r.orig}" alt="orig" loading="lazy">
    </div>
    <div class="img-col">
      <label>异常热力图 (Grad-CAM)</label>
      <img src="/img/${r.heat}" alt="heatmap" loading="lazy">
      <div class="legend">
        <div class="leg-row"><div class="leg-dot" style="background:#ef4444"></div>红色 = 黑色素瘤</div>
        <div class="leg-row"><div class="leg-dot" style="background:#3b82f6"></div>蓝色 = 皮肤癌</div>
        <div class="leg-row"><div class="leg-dot" style="background:#22c55e"></div>绿色 = 非癌症/健康</div>
      </div>
    </div>
    <div class="prob-col">
      <h4>各类别概率</h4>
      ${bars}
      <div class="pred-box">
        <div class="pcls" style="color:${r.risk_color}">${r.risk_cn}</div>
        <div class="pconf">主预测置信度 ${(r.pred_prob*100).toFixed(1)}%</div>
      </div>
      <a class="download-link" href="${repUrl}" target="_blank">📄 查看风险报告</a>
    </div>
  </div>
</div>`;
}

function errCard(r){
  return `<div class="rcard">
  <div class="rcard-head">
    <span>❌</span>
    <span class="fname">${r.filename}</span>
    <span class="rbadge rbHIGH_MEL">失败</span>
  </div>
  <div class="err-body">错误: ${r.error}</div>
</div>`;
}

// ── Helpers ───────────────────────────────────────────────────
async function get(url){ const r=await fetch(url); return r.json(); }
async function post(url){ const r=await fetch(url,{method:"POST"}); return r.json(); }

function toast(msg, type="ok"){
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.className   = "toast "+type+" show";
  setTimeout(()=>t.classList.remove("show"), 4000);
}
</script>
</body>
</html>"""

# ══════════════════════════════════════════════════════════════
# 7. 入口
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("   皮肤癌 AI 风险评估系统")
    print(f"   Device : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    print("=" * 55)

    if load_model():
        print(f"   ✅  模型已加载: {MODEL_PATH}")
    else:
        print(f"   ⚠️   找不到 {MODEL_PATH}，推理功能不可用")
        print("        请先运行 train.py 完成训练")

    ORG_DIR.mkdir(exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)
    print(f"   📂  输入: {ORG_DIR}/   输出: {OUT_DIR}/")
    print()
    print("   🌐  浏览器打开: http://localhost:5000")
    print("   🛑  按 Ctrl+C 停止服务")
    print("=" * 55)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)