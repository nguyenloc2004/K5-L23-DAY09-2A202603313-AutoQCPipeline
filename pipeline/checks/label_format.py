"""CHECK 2 — Dòng nhãn YOLO: sai định dạng, ID lớp lạ, box tràn ảnh, box rỗng, box lặp.   (Script 2 trong slide)   ★★

Test:  pytest tests/test_label_format.py -q

Mỗi dòng nhãn: `id cx cy w h` (5 số, toạ độ 0–1). Trả về list[Issue]; `Issue.line` = số dòng (đếm từ 1).

| code             | severity | khi nào                                                                  |
|------------------|----------|--------------------------------------------------------------------------|
| BAD_FORMAT       | ERROR    | không đúng 5 cột số (dùng `parse_yolo_line`, bắt `ValueError`)           |
| BAD_CLASS_ID     | ERROR    | id lớp ngoài 0..len(ds.classes)-1                                        |
| BOX_ZERO         | ERROR    | w <= 0 hoặc h <= 0                                                       |
| BOX_OUT_OF_IMAGE | ERROR    | mép box vượt [0,1] quá `cfg["label_format"]["overflow_tol"]`; extra={"overflow": mức tràn} |
| DUP_BOX          | ERROR    | cùng lớp với một dòng TRƯỚC ĐÓ và IoU > `dup_box_iou`; báo ở dòng SAU    |
| BOX_TINY         | WARN     | cạnh ngắn của box (px) < `min_box_px`  (cần `ds.image_size(stem)`)       |
| BOX_HUGE         | WARN     | w*h > `max_box_frac`                                                     |

Gợi ý
- Mép trái = x - w/2, mép phải = x + w/2 (xem câu hỏi "dòng YOLO nào lỗi?" trong slide).
- Một dòng hỏng (BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO) thì `continue`, đừng kiểm tiếp hình học của dòng đó.
- Dòng thiếu cột làm script của slide dừng giữa chừng: đây là lý do phải `try/except`.
- `iou_xywh(a, b)` có sẵn trong common.py.
- Ngưỡng nằm trong `cfg["label_format"]` — đừng viết cứng số vào code.
"""
from ..common import ERROR, WARN, Issue, iou_xywh, parse_yolo_line  # noqa: F401

NAME = "label_format"


def run(ds, cfg):
    p = cfg["label_format"]
    tol, min_px, max_frac, dup_iou = p["overflow_tol"], p["min_box_px"], p["max_box_frac"], p["dup_box_iou"]
    n_classes = len(ds.classes)
    issues = []

    for lf in ds.label_files():
        stem = lf.stem
        size = ds.image_size(stem)                      # (W, H) px; None nếu không có ảnh / ảnh hỏng
        kept = []                                       # (số dòng, box) các dòng hợp lệ đã gặp — để tìm DUP_BOX
        for no, text in ds.read_label_lines(stem) or []:   # số dòng gốc, đã bỏ dòng trống
            # 1) định dạng: đúng 5 cột số, không nan/inf
            try:
                c, x, y, w, h = parse_yolo_line(text)
            except ValueError as e:
                issues.append(Issue(NAME, ERROR, "BAD_FORMAT", stem, f"dòng sai định dạng ({e}): {text.strip()!r}", no))
                continue
            # 2) id lớp hợp lệ
            if not 0 <= c < n_classes:
                issues.append(Issue(NAME, ERROR, "BAD_CLASS_ID", stem, f"id lớp {c} ngoài 0..{n_classes - 1}", no, {"class_id": c}))
                continue
            # 3) box có diện tích
            if w <= 0 or h <= 0:
                issues.append(Issue(NAME, ERROR, "BOX_ZERO", stem, f"box rộng/cao bằng 0 (w={w}, h={h})", no))
                continue

            # 4) tràn ảnh: mép nào vượt [0,1] nhiều nhất
            overflow = max(0.0, -(x - w / 2), (x + w / 2) - 1, -(y - h / 2), (y + h / 2) - 1)
            if overflow > tol:
                issues.append(Issue(NAME, ERROR, "BOX_OUT_OF_IMAGE", stem, f"box tràn ảnh {overflow:.4f}", no,
                                    {"overflow": round(overflow, 6)}))

            # 5) box lặp: cùng lớp với một dòng TRƯỚC, IoU > ngưỡng -> báo ở dòng sau
            box = (x, y, w, h)
            dup = next((pn for pn, (pc, pb) in kept if pc == c and iou_xywh(pb, box) > dup_iou), None)
            if dup is not None:
                issues.append(Issue(NAME, ERROR, "DUP_BOX", stem, f"trùng box với dòng {dup}", no, {"of_line": dup}))
            kept.append((no, (c, box)))

            # 6) cảnh báo kích thước (cần kích thước ảnh, ảnh không mở được thì bỏ qua BOX_TINY)
            if size:
                short_px = min(w * size[0], h * size[1])
                if short_px < min_px:
                    issues.append(Issue(NAME, WARN, "BOX_TINY", stem, f"cạnh ngắn {short_px:.1f}px < {min_px}px", no,
                                        {"short_px": round(short_px, 2)}))
            if w * h > max_frac:
                issues.append(Issue(NAME, WARN, "BOX_HUGE", stem, f"box chiếm {w * h:.2%} ảnh", no, {"area_frac": round(w * h, 4)}))
    return issues
