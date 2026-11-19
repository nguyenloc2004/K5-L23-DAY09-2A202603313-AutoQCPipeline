"""CHECK 6 — Ý kiến thứ hai: so nhãn với một model LỚN HƠN (yolo26l).   ★★★ (khó nhất, đáng điểm nhất)

Test:  pytest tests/test_second_opinion.py -q

Máy kiểm quy tắc, người kiểm ý nghĩa (slide). Các lỗi như thiếu nhãn / gán nhầm lớp hợp lệ / box lỏng
KHÔNG bắt được bằng quy tắc cú pháp. Ta nhờ một model mạnh hơn "nhìn lại" và CHỈ CẢNH BÁO (model cũng có thể sai):
những ảnh bị cờ sẽ vào hàng đợi cho NGƯỜI duyệt.

`preds[stem]` = list `[id_lớp, conf, cx, cy, w, h]` của model (lấy qua `get_preds(ds, cfg, "checker")` nếu preds=None).
`L = ds.valid_boxes(stem)` = list `(id_lớp, cx, cy, w, h)` của nhãn (đã bỏ dòng hỏng).
Ngưỡng ở `cfg["second_opinion"]`: conf, min_area_px, fp_conf, iou_missing, iou_match, iou_loose.

Thuật toán gợi ý, với từng ảnh (bỏ qua ảnh không có trong preds hoặc không có file nhãn):
  P = dự đoán có conf >= conf  VÀ diện tích px (w*W * h*H) >= min_area_px
  Với mỗi dự đoán r trong P: lấy các nhãn có IoU(r, nhãn) >= iou_missing  (cands)
   - không có cands                            -> POSSIBLE_MISSING     (model thấy vật mà nhãn không có box)
   - có cands cùng lớp, tốt nhất IoU < iou_loose -> POSSIBLE_LOOSE_BOX  (đúng vật, box lỏng/lệch)
   - không có cands cùng lớp, nhưng cands tốt nhất IoU >= iou_match (khác lớp) -> POSSIBLE_WRONG_CLASS
  Với mỗi dòng nhãn hợp lệ: nếu IoU với MỌI dự đoán (conf >= fp_conf, không lọc diện tích) đều < iou_missing
                                             -> POSSIBLE_FALSE_POS  (box nhãn không khớp vật nào model thấy; Issue.line = số dòng)

Chi tiết các ca biên (test kiểm đúng các ca này)
- `cands` gồm nhãn MỌI lớp có IoU >= iou_missing; "tốt nhất cùng lớp" = cands cùng lớp có IoU lớn nhất; "tốt nhất khác lớp" tương tự với cands khác lớp.
- Thứ tự quyết định cho mỗi dự đoán: (1) cands rỗng -> MISSING. (2) có cands cùng lớp: IoU tốt nhất < iou_loose -> LOOSE_BOX, ngược lại KHÔNG cờ
  (nhãn đúng lớp đã phủ vật này, kể cả khi còn nhãn khác lớp chồng lên). (3) không có cands cùng lớp: nếu IoU tốt nhất khác lớp >= iou_match -> WRONG_CLASS;
  nếu chỉ có khác lớp với IoU trong [iou_missing, iou_match) -> không cờ gì (mơ hồ: để cho FALSE_POS / lần duyệt khác).
- Một dự đoán sinh tối đa MỘT issue (MISSING / LOOSE_BOX / WRONG_CLASS); một ảnh có thể có nhiều issue.
- Thứ tự issue: theo `ds.image_files()` rồi theo thứ tự dự đoán trong `preds[stem]`, cuối cùng là các FALSE_POS theo số dòng; `Issue.image` = stem, `Issue.line` chỉ đặt cho FALSE_POS.

Lưu ý số dòng: dùng `ds.read_label_lines(stem)` (giữ số dòng gốc, kể cả dòng hỏng/trống) để lấy `Issue.line`; `valid_boxes` KHÔNG giữ số dòng.

Tất cả severity WARN; `Issue.extra` nên có {"conf": ...} hoặc {"iou": ...} (hàng đợi duyệt dùng conf để xếp ưu tiên).

Gợi ý
- `iou_xywh` có sẵn. Dòng nhãn hỏng không được làm script chết (dùng `ds.valid_boxes` và `parse_yolo_line` trong try).
- Kết quả phụ thuộc mạnh vào ngưỡng: calibrate chúng ở Bước 3 thay vì đoán.
"""
from ..common import WARN, Issue, iou_xywh, parse_yolo_line
from ..model_cache import get_preds

NAME = "second_opinion"


def _labels_with_lines(ds, stem):
    """[(số dòng gốc, (id, cx, cy, w, h))] các dòng parse được và id lớp hợp lệ (như valid_boxes nhưng giữ số dòng)."""
    out = []
    for no, text in ds.read_label_lines(stem) or []:
        try:
            b = parse_yolo_line(text)
        except ValueError:
            continue                                   # dòng hỏng: label_format báo, ở đây bỏ qua
        if 0 <= b[0] < len(ds.classes):
            out.append((no, b))
    return out


def run(ds, cfg, preds=None):
    p = cfg["second_opinion"]
    conf_min, min_area, fp_conf = p["conf"], p["min_area_px"], p["fp_conf"]
    iou_missing, iou_match, iou_loose = p["iou_missing"], p["iou_match"], p["iou_loose"]
    if preds is None:
        preds = get_preds(ds, cfg, "checker")

    issues = []
    for img in ds.image_files():
        stem = img.stem
        if stem not in preds or ds.read_label_lines(stem) is None:    # không có dự đoán / chưa có file nhãn
            continue
        size = ds.image_size(stem)
        if size is None:                                               # ảnh hỏng: check images báo
            continue
        W, H = size
        labels = _labels_with_lines(ds, stem)
        rows = preds[stem]

        # A) Từng dự đoán tin cậy, đủ lớn -> MISSING / LOOSE_BOX / WRONG_CLASS (tối đa 1 issue mỗi dự đoán)
        for r in rows:
            rc, rconf, rbox = int(r[0]), float(r[1]), tuple(r[2:6])
            if rconf < conf_min or rbox[2] * W * rbox[3] * H < min_area:
                continue
            cands = [(iou_xywh(rbox, lb[1:]), lb[0]) for _, lb in labels]
            cands = [(iou, lc) for iou, lc in cands if iou >= iou_missing]
            if not cands:
                issues.append(Issue(NAME, WARN, "POSSIBLE_MISSING", stem,
                                    f"model thấy {ds.classes[rc]} (conf {rconf:.2f}) mà nhãn không có box",
                                    extra={"conf": rconf, "pred_class": rc, "box": list(rbox)}))
                continue
            same = [iou for iou, lc in cands if lc == rc]
            if same:
                best = max(same)
                if best < iou_loose:
                    issues.append(Issue(NAME, WARN, "POSSIBLE_LOOSE_BOX", stem,
                                        f"box {ds.classes[rc]} lệch/lỏng so với model (IoU {best:.2f})",
                                        extra={"conf": rconf, "iou": round(best, 4)}))
                continue                                               # nhãn đúng lớp đã phủ vật này
            best_iou, best_cls = max(cands)
            if best_iou >= iou_match:
                issues.append(Issue(NAME, WARN, "POSSIBLE_WRONG_CLASS", stem,
                                    f"nhãn {ds.classes[best_cls]} nhưng model nói {ds.classes[rc]} (conf {rconf:.2f}, IoU {best_iou:.2f})",
                                    extra={"conf": rconf, "iou": round(best_iou, 4), "label_class": best_cls, "pred_class": rc}))
            # IoU khác lớp trong [iou_missing, iou_match): mơ hồ -> không cờ

        # B) Từng dòng nhãn không khớp dự đoán nào (conf >= fp_conf, KHÔNG lọc diện tích) -> FALSE_POS
        fp_boxes = [tuple(r[2:6]) for r in rows if float(r[1]) >= fp_conf]
        for no, lb in labels:
            best = max((iou_xywh(pb, lb[1:]) for pb in fp_boxes), default=0.0)
            if best < iou_missing:
                issues.append(Issue(NAME, WARN, "POSSIBLE_FALSE_POS", stem,
                                    f"box {ds.classes[lb[0]]} không khớp vật nào model thấy (IoU tốt nhất {best:.2f})",
                                    no, {"iou": round(best, 4)}))
    return issues
