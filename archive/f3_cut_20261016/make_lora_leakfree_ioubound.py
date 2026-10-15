"""生成 F3 训练脚本副本（train_lora_leakfree_ioubound.py）的加工脚本。

保留它是为了**可复现 provenance**：任何人可以重跑本脚本，从原件重新生成副本，
并验证 backups/patches/lora_leakfree_ioubound_copy_*.diff 与之一致。

用法：python backups/patches/make_lora_leakfree_ioubound.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "train_lora_leakfree.py"
DST = ROOT / "train_lora_leakfree_ioubound.py"

DOC_OLD = '''用法：
    python train_lora_leakfree.py --epochs 1     # 冒烟测试
    python train_lora_leakfree.py --epochs 20    # 快速
    python train_lora_leakfree.py                # 完整(默认 150)
"""'''

DOC_NEW = '''用法：
    python train_lora_leakfree_ioubound.py --epochs 40
"""'''

ADD_DOC = '''
【F3：本文件是 train_lora_leakfree.py 的**副本**，用于落实申报书点名的损失函数】—— 原件一字未动。

差异（**唯一实质改动 = 损失组合**）：
    原: focal_loss + dice_loss
    本: focal_loss + iou_loss + lambda*boundary_loss   (lambda=1.0，预注册固定)

[!] 关于"只改一行"的更正：损失**不在**本脚本里，而在被 import 的
    `train_adapter_v4.train_step` 内。故**无法"改一行"** —— 本副本的做法是
    **在文件内重写一份 `train_step_ioubound`**，逐行照抄原序列、**只替换损失行**，
    并复用 `train_adapter_v4` 的 `transform/sample_points/gt_bbox/focal_loss`。
    **不改 `train_adapter_v4.py`**（它被主结果依赖）。

[!] 两条硬护栏（审查强制，防静默错误）：
    1. 随机翻转**只用一次** `random.random()`（同原件），**不新增任何 random/np.random 抽取**
       —— 否则 RNG 流整体位移，"唯一变量=损失"不成立。
    2. 边界损失的距离图在 `train_step` 内对**已翻转+已 resize 的 gt256 现算**，
       **不预计算、不缓存** —— 预计算会与随机翻转**错位**（静默错误，日志看不出）。

声明：本实现是**按申报书原理实现、非逐行复现**（申报书只给了损失名称、未给公式），同 DTICT 的处理。

【F3 预注册（开跑前定死，此后不得再改）】
    lambda = 1.0 固定 | 保留 Focal | 距离图**有符号** | 严格 **40 轮**（与主结果同轮数才可横比）
    拆分声明：本次是**复合替换**（区域项 Dice→IoU 近乎等价改写 + **新增 Boundary**），
    报告**不得**把结果归因到 IoU 或 Boundary 任一子项。
'''

NEW_STEP = '''def train_step_ioubound(sam, img_rgb, gt, opt, scaler):
    """与 `train_adapter_v4.train_step` **逐行相同**，**唯一差异是最后那行损失组合**。

    原 : loss = focal_loss(...) + dice_loss(...)
    本 : loss = combined_loss(...) = Focal + IoU + lambda*Boundary

    [护栏] RNG 抽取次数与顺序**与原件完全一致**（翻转 1 次、点/框选择 1 次）——
           据此"唯一变量 = 损失"才成立。**不得**在此函数内新增任何 random/np.random 调用。
    [护栏] 距离图在函数内、对已翻转+已 resize 的 gt256 **现算**。
    """
    h, w = gt.shape
    if random.random() < 0.5:                      # [RNG #1] 同原件
        img_rgb = img_rgb[:, ::-1, :].copy()
        gt = gt[:, ::-1].copy()

    img1024 = transform.apply_image(img_rgb)
    img_t = torch.as_tensor(img1024, dtype=torch.float32).permute(2, 0, 1)[None].to(device)

    if random.random() < 0.7:                      # [RNG #2] 同原件
        coords, labels = sample_points(gt)
        if coords is None:
            return None
        c1024 = transform.apply_coords(coords, (h, w))
        c_t = torch.as_tensor(c1024, dtype=torch.float32)[None].to(device)
        l_t = torch.as_tensor(labels, dtype=torch.int64)[None].to(device)
        points, boxes = (c_t, l_t), None
    else:
        box = gt_bbox(gt)
        if box is None:
            return None
        b1024 = transform.apply_boxes(box, (h, w))
        b_t = torch.as_tensor(b1024, dtype=torch.float32)[None].to(device)
        points, boxes = None, b_t

    x = sam.preprocess(img_t)
    with torch.cuda.amp.autocast():
        img_emb = sam.image_encoder(x)  # 不 no_grad：LoRA 需要梯度
        sparse, dense = sam.prompt_encoder(points=points, boxes=boxes, masks=None)
        low_res, _ = sam.mask_decoder(
            image_embeddings=img_emb, image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            multimask_output=False,
        )
    gt256 = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_NEAREST)
    gt_t = torch.as_tensor((gt256 > 0), dtype=torch.float32)[None, None].to(device)
    # ===== 唯一改动：损失组合（在 autocast 之外 => fp32，防 IoU 的 sum(p) 在 fp16 溢出）=====
    loss = combined_loss(low_res, gt_t, focal_loss)
    # ===== 以上为唯一改动 =====
    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    return loss.item()


def main():'''

ASSERT_BLOCK = '''    # [F3] 继承 seed 副本的划分硬断言：保证与主结果**同划分**（横比前提）
    if len(test_files) != 92 or n_te != 21:
        raise RuntimeError(f"划分异常: test={len(test_files)}/{n_te}原图（应 92/21），拒绝产出不可比结果")
    _mcsv = RES_DIR / "lora_leakfree_test_per_image.csv"
    if _mcsv.exists():
        with open(_mcsv, newline="", encoding="utf-8-sig") as _fh:
            _nm = {r["name"] for r in csv.DictReader(_fh) if r.get("name")}
        _nh = {Path(f).name for f in test_files}
        if _nm != _nh:
            raise RuntimeError(f"划分与原主结果不一致（{len(_nh)} vs {len(_nm)}），拒绝产出不可比结果")

    log(f"Train412 {len(files)} 张 / {n_tr + n_va + n_te} 张原始大图")'''

HEADER_BLOCK = '''    log(f"# epochs={args.epochs} seed=0  口径=框提示（选择与报告一致）")
    log(f"# [F3] 损失 = Focal + IoU + {LAMBDA_BOUNDARY}*Boundary（有符号距离图）；按原理实现、非逐行复现申报书")
    log(f"# [F3] 预注册：lambda={LAMBDA_BOUNDARY} 固定 / 保留 Focal / 有符号 / 严格 40 轮 / 单 seed")
    log(f"# argv: {' '.join(sys.argv)}")
    for _f in ["train_lora_leakfree_ioubound.py", "losses_iou_boundary.py",
               "train_adapter_v4.py", "train_lora_leakfree.py"]:
        try:
            _h = hashlib.sha256((BASE / _f).read_bytes()).hexdigest()[:16]
            log(f"# sha256[:16] {_f} = {_h}")
        except Exception as _e:
            log(f"# sha256 失败 {_f}: {_e}")'''


def rep(s, old, new, n=1):
    assert s.count(old) == n, f"匹配 {s.count(old)} 次（应 {n}）: {old[:60]!r}"
    return s.replace(old, new)


def main():
    s = SRC.read_text(encoding="utf-8")

    # 1) 文档头
    s = rep(s, DOC_OLD, DOC_NEW)
    s = rep(s, '"""\nimport argparse\n', '"""\n' + ADD_DOC + '\nimport argparse\n')

    # 2) 追加 import（纯新增，不动原行）
    s = rep(s,
            "from train_adapter_v4 import apply_lora, lora_params, train_step, LR, LR_MIN, CKPT, AISD\n",
            "from train_adapter_v4 import apply_lora, lora_params, train_step, LR, LR_MIN, CKPT, AISD\n"
            "# [F3] 追加：重写 train_step 所需的工具函数（复用，不重写逻辑）\n"
            "from train_adapter_v4 import transform, sample_points, gt_bbox, focal_loss\n"
            "from losses_iou_boundary import combined_loss, LAMBDA_BOUNDARY\n")

    # 3) import sys/hashlib
    s = rep(s, "import argparse\nimport csv\nimport math\nimport random\nimport time\n",
            "import argparse\nimport csv\nimport hashlib\nimport math\nimport random\nimport sys\nimport time\n")

    # 4) 输出路径加后缀
    s = rep(s, 'OUT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree.pth")',
            'OUT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree_ioubound.pth")')
    s = rep(s, 'RESUME_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree_resume.pth")',
            'RESUME_CKPT = str(BASE / "checkpoints" / "sam_vit_b_shadow_v4_lora_leakfree_ioubound_resume.pth")')

    # 5) 日志名
    s = rep(s, 'log_path = LOG_DIR / f"lora_leakfree_{ts}.log"',
            'log_path = LOG_DIR / f"lora_leakfree_ioubound_{ts}.log"')

    # 6) 结果文件名
    for a, b in [('RES_DIR / "lora_leakfree_test_per_image.csv"', 'RES_DIR / "lora_leakfree_ioubound_test_per_image.csv"'),
                 ("RES_DIR / 'lora_leakfree_test_per_image.csv'", "RES_DIR / 'lora_leakfree_ioubound_test_per_image.csv'"),
                 ('RES_DIR / "lora_leakfree_test.csv"', 'RES_DIR / "lora_leakfree_ioubound_test.csv"'),
                 ("RES_DIR / 'lora_leakfree_test.csv'", "RES_DIR / 'lora_leakfree_ioubound_test.csv'")]:
        s = s.replace(a, b)

    # 7) 插入重写的 train_step
    s = rep(s, "def main():", NEW_STEP)

    # 8) 调用点
    s = rep(s, "            loss = train_step(sam, img, gt, opt, scaler)",
            "            loss = train_step_ioubound(sam, img, gt, opt, scaler)   # [F3] 唯一改动点")

    # 9) 划分硬断言
    s = rep(s, '    log(f"Train412 {len(files)} 张 / {n_tr + n_va + n_te} 张原始大图")', ASSERT_BLOCK)

    # 10) 日志头哈希与 argv
    s = rep(s, '    log(f"# epochs={args.epochs} seed=0  口径=框提示（选择与报告一致）")', HEADER_BLOCK)

    DST.write_text(s, encoding="utf-8")
    print("[OK] 已生成", DST.name)


if __name__ == "__main__":
    main()
