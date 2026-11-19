"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict

import yaml

from ..common import ERROR, parse_yolo_line

DROP_CODES = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
REMOVE_CODES = {"LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}          # + mọi mã bắt đầu bằng DUP_IMAGE_


def _fx(image, line, action, before, after):
    return dict(image=image, line=line, action=action, before=before, after=after)


def _is_remove(code):
    return code in REMOVE_CODES or code.startswith("DUP_IMAGE_")


def _clip(text):
    """Cắt box của một dòng về [0,1]. Trả dòng mới, hoặc None nếu sau khi cắt box không còn diện tích."""
    c, x, y, w, h = parse_yolo_line(text)
    x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
    x2, y2 = min(1.0, x + w / 2), min(1.0, y + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    nx, ny, nw, nh = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return f"{c} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"


def fix(ds, issues, cfg):
    errors = [i for i in issues if i.severity == ERROR]          # WARN là việc của người: không đụng
    fixes = []

    # 1) Xoá ảnh hỏng / trùng / rò rỉ (cùng file nhãn của nó)
    removed = set()
    for i in errors:
        if not _is_remove(i.code) or i.image in removed:
            continue
        img = ds.image_path(i.image)
        lab = ds.label_path(i.image)
        if img is None and not lab.exists():                     # đã xoá ở lần chạy trước: idempotent
            continue
        if img is not None:
            img.unlink()
        if lab.exists():
            lab.unlink()
        removed.add(i.image)
        fixes.append(_fx(i.image, "", "remove_image", img.name if img else lab.name, f"(đã xoá: {i.code})"))

    # 2) Ảnh chưa có nhãn: KHÔNG bịa nhãn, chỉ ghi lại để chuyển người gán
    for i in errors:
        if i.code == "MISSING_LABEL_FILE" and i.image not in removed and not ds.label_path(i.image).exists():
            fixes.append(_fx(i.image, "", "needs_label", "(không có file)", "chuyển người gán"))

    # 3) Tên lớp sai trong data.yaml: ghi lại names đúng cfg["classes"] (một lần dù có nhiều issue)
    if any(i.code == "BAD_CLASS_NAME" for i in errors):
        data = {}
        if ds.yaml_path.exists():
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
        old = ds.yaml_names()
        new = {k: v for k, v in enumerate(cfg["classes"])}
        if old != new:
            data["names"] = new
            ds.yaml_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
            fixes.append(_fx("data.yaml", "", "rename_classes", str(list(old.values())), str(list(new.values()))))

    # 4) Lỗi dòng nhãn: gom theo file, dùng số dòng GỐC, ghi lại file MỘT lần
    drop, clip = defaultdict(set), defaultdict(set)
    for i in errors:
        if i.line is None:
            continue
        if i.code in DROP_CODES:
            drop[i.image].add(i.line)
        elif i.code == "BOX_OUT_OF_IMAGE":
            clip[i.image].add(i.line)

    for stem in sorted(set(drop) | set(clip)):
        path = ds.label_path(stem)
        if stem in removed or not path.exists():                 # ảnh đã bị xoá: bỏ qua êm
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()   # giữ cả dòng trống -> số dòng gốc
        out, changed = [], False
        for no, text in enumerate(lines, 1):
            if no in drop[stem]:                                 # xoá dòng thắng clip (dòng có 2 lỗi chỉ ghi 1 fix)
                fixes.append(_fx(stem, no, "drop_line", text, ""))
                changed = True
                continue
            if no in clip[stem]:
                try:
                    new = _clip(text)
                except ValueError:                               # dòng hỏng lẽ ra đã bị drop; phòng hờ: xoá
                    new = None
                changed = True
                if new is None:                                  # box nằm hẳn ngoài ảnh: xoá, không ghi box rỗng
                    fixes.append(_fx(stem, no, "drop_line", text, ""))
                    continue
                fixes.append(_fx(stem, no, "clip_box", text, new))
                out.append(new)
                continue
            out.append(text)
        if changed:
            path.write_text("".join(l + "\n" for l in out), encoding="utf-8")
    return fixes
