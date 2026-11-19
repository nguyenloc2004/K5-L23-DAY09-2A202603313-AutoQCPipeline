# Báo cáo Day 09 — Nguyễn Hữu Lộc, 2A202603313

## 1. Kết luận trước
**Kết luận: Hold.** Dataset đã sạch lỗi bắt buộc nhưng **chưa đủ điều kiện phát hành**.

- **Số ảnh:** vào 224 ảnh, loại 22 ảnh hỏng/trùng/rò rỉ test, còn **202 ảnh** trong dataset.
- **Lỗi bắt buộc:** còn **0** (máy đã tự sửa, người đã gán nhãn cho ảnh thiếu nhãn).
- **Vì sao Hold:** còn **38/202 ảnh (18.8%)** có cảnh báo mà chưa ai xem, vượt ngưỡng 10%. Lý do là máy nghi ngờ 125 ảnh, nhưng người duyệt chỉ xem được 80 ảnh.

## 2. Lỗi và phân việc
Từ `02_error_list_v1` (trước khi sửa), gom theo mã lỗi:

| Mã lỗi | Số ảnh/dòng | Ai sửa (máy/người) | Vì sao |
|---|---:|---|---|
| DUP_IMAGE_EXACT | 8 ảnh | máy (loại ở bước 1) | md5 trùng từng byte: chắc chắn |
| DUP_IMAGE_NEAR | 4 ảnh | máy (loại ở bước 1) | phash ≤ 12; đã mở xem cả 4 cặp, đều trùng thật |
| LEAK_TEST | 5 ảnh | máy (loại ở bước 1) | trùng ảnh tập test: giữ lại thì điểm test bị thổi phồng |
| BAD_SIZE / CORRUPT_IMAGE | 3 / 2 ảnh | máy (loại ở bước 1) | ảnh quá nhỏ / không mở được |
| BAD_CLASS_ID | 6 dòng | máy (`drop_line`) | id 93, 4, −1… không thuộc lớp nào |
| BAD_FORMAT | 3 dòng | máy (`drop_line`) | thiếu cột / `nan` |
| BOX_ZERO | 4 dòng | máy (`drop_line`) | box rộng/cao 0 |
| DUP_BOX | 7 dòng | máy (`drop_line`) | cùng lớp, IoU > 0.9 với dòng trước |
| BOX_OUT_OF_IMAGE | 8 dòng | máy (`clip_box`) | cắt về [0,1], giữ lớp |
| BAD_CLASS_NAME | 2 issue (data.yaml) | máy (`rename_classes`) | tên lớp viết hoa sai |
| MISSING_LABEL_FILE | 6 ảnh | **người** (`needs_label`) | máy không được bịa nhãn |
| POSSIBLE_MISSING | 52 ảnh | **người** | model lớn thấy vật mà nhãn không có |
| POSSIBLE_WRONG_CLASS | 18 ảnh | **người** | model nói lớp khác, IoU ≥ 0.5 |
| POSSIBLE_LOOSE_BOX | 24 ảnh | **người** | đúng vật nhưng box lệch/lỏng |
| POSSIBLE_FALSE_POS | 27 ảnh | **người** | box nhãn không khớp vật nào model thấy |
| EMPTY_LABEL / BOX_TINY / BOX_HUGE | 12 / 4 / 3 ảnh | **người** | cảnh báo, máy không tự quyết |

Máy sửa **35 thay đổi** (`03_fixes.csv`: 20 `drop_line`, 8 `clip_box`, 1 `rename_classes`, 6 `needs_label`); sau sửa còn đúng 6 lỗi bắt buộc = 6 ảnh chưa có nhãn.

**Thứ tự hàng đợi** (`05_review_queue.csv`): điểm = tổng trọng số mã lỗi × (1 + conf của model). `NEEDS_LABEL` có trọng số 10 nên 6 ảnh chưa có nhãn nằm ở hạng 2–7; hạng 1 là `img_0020` (điểm 11.5: vừa bị máy xoá dòng, vừa có MISSING + WRONG_CLASS). Sau đó là các ảnh có `POSSIBLE_WRONG_CLASS` (trọng số 3) và `POSSIBLE_MISSING` (2) với conf cao.
**Ngân sách 80 ảnh**: người duyệt 80/125 ảnh. Trong 80 ảnh đó có 50 ảnh mang `POSSIBLE_MISSING`, 18 `POSSIBLE_WRONG_CLASS` (tức **toàn bộ** 18 ảnh nghi nhầm lớp), 12 `POSSIBLE_FALSE_POS`, 8 `POSSIBLE_LOOSE_BOX`. Sau duyệt, số ảnh còn MISSING giảm 52 → 12 và WRONG_CLASS 18 → 3 (`07_error_list_v3`). 45 ảnh còn lại không ai xem, phần lớn là FALSE_POS / LOOSE_BOX trọng số thấp.

## 3. Calibration (2 ngưỡng)
Số trên bộ calib (39 ảnh, `submission/calibration/calibration_report.md`):

| Tham số | Khởi đầu | Đã chọn | Precision / Recall (trước → sau) | Lý do (đánh đổi) |
|---|---|---|---|---|
| `duplicates.phash_dist` | 2 | **12** | DUP_IMAGE_NEAR: 1.00 / 0.50 → 1.00 / 0.67 (EXACT giữ 1.00 / 1.00) | Ảnh bị cờ sẽ **bị xoá** → ưu tiên precision. 10–14 cho kết quả giống hệt nhau trên cả calib và raw; chọn 12 ở giữa khoảng ổn định, xa vùng 16 |
| `second_opinion.conf` | 0.25 | **0.6** | MISSING: 0.50 / 1.00 → 0.57 / 1.00; WRONG_CLASS: 0.56 / 0.83 → 0.62 / 0.83; LOOSE_BOX: 0.29 / 0.40 → 0.20 / 0.20 | Cảnh báo chỉ tốn công người xem → ưu tiên recall. 0.6 là mức cao nhất còn giữ recall MISSING = 1.00. Cái giá: LOOSE_BOX tụt (calib chỉ có 5 lỗi loại này) |

Trên raw, `conf` 0.25 → 0.6 làm hàng đợi giảm 143 → 125 ảnh và tỉ lệ cảnh báo chưa duyệt 28.7% → 18.8%. Tôi **không** chọn 0.8, dù 0.8 là mức duy nhất cho Release (7.9%): ở 0.8 recall MISSING trên calib chỉ còn 0.75 (bỏ sót 2/8). Trên raw, `img_0030` (một người bên phải bị thiếu box, tôi tự thấy ở Bước 1) mất cờ MISSING và tụt từ hạng 21 xuống hạng 56 trong hàng đợi. Đó sẽ là một "Release giả".

**Chỗ calibration có thể đánh lừa:** trên calib, `phash_dist = 16` có F1 tốt nhất cho DUP_IMAGE_NEAR (0.91). Nhưng chạy trên raw, từ 14 lên 16 số `LEAK_TEST` nhảy **5 → 11** và có thêm 3 cặp NEAR mới có khoảng cách **đúng bằng 16** (img_0011/0008, img_0084/0074, img_0177/0113): rất có thể là gộp nhầm ảnh khác nhau. Tổng ảnh bị loại sẽ là 26 thay vì 17. Ngoài ra `phash_dist = 20` còn làm recall DUP_IMAGE_EXACT tụt về 0.33, vì ảnh trùng byte bị báo thành LEAK_TEST (mã ưu tiên cao hơn). Với 12, tôi đã mở xem cả 4 cặp DUP_IMAGE_NEAR (img_0076/0042, img_0143/0037, img_0186/0061, img_0223/0077) và 5 ảnh LEAK_TEST: tất cả đều trùng thật.

## 4. Cửa chặn và điều pipeline KHÔNG bắt được
**(a) Gate chặn đúng.** Chạy `python -m pipeline.run --budget 0 --reports work/rep_nb --export-dir work/exp_nb` (không có người duyệt): kết luận **REWORK**, "còn 6 lỗi bắt buộc (ERROR)" (6 ảnh `MISSING_LABEL_FILE` không ai gán), **mã thoát 20**. Log in "KHÔNG xuất: còn lỗi bắt buộc (Rework)" và thư mục `work/exp_nb` không được tạo.

**(b) Điểm mù có bằng chứng.**
1. **Box thừa mà model không phản đối:** ở Bước 1 tôi thấy `img_0002` có một box person thừa. Pipeline không cờ gì cho ảnh này ở cả v1, v2 và v3, nên ảnh **không vào hàng đợi** và lỗi lọt vào dataset xuất ra.
2. **Box lỏng/lệch nhẹ:** `POSSIBLE_LOOSE_BOX` trên calib chỉ đạt precision 0.20 / recall 0.20 (1/5 lỗi chèn). Sau duyệt, raw vẫn còn 20 ảnh mang cờ này. Checker gần như không đáng tin cho loại lỗi này.
3. **Ảnh trùng biến đổi mạnh:** trên calib, DUP_IMAGE_NEAR chỉ bắt 4/6 ảnh ở mọi `phash_dist` từ 4 đến 14. 2 ảnh bị biến đổi nhiều không ngưỡng an toàn nào bắt được. Muốn bắt chúng phải nâng lên 16, và khi đó lại xoá nhầm trên raw (mục 3).
4. **Nhãn thật không trùng mẫu lỗi chèn:** calib dựng từ nhãn người, raw từ nhãn máy. Nhầm lớp trong calib chủ yếu là person→car, còn raw nghiêng về car/truck/bus. Recall trên calib vì vậy chưa phải recall trên raw.

## 5. Train: raw vs clean (tùy chọn, chấm riêng)
Chưa chạy. Tôi ưu tiên hoàn thành phần lõi (pipeline, calibration, báo cáo) trước; phần train trên Colab để làm sau nếu còn thời gian.

## 6. Quyết định và ghi chú AI
**(a) Quyết định:** giữ **Hold**. Dataset `clean_v1` không còn lỗi bắt buộc nhưng **chưa phát hành**. Để thành Release cần một vòng duyệt nữa cho khoảng 38–45 ảnh còn cảnh báo (chủ yếu FALSE_POS / LOOSE_BOX): chạy `--no-review`, người sửa nhãn rồi `--resume-after-review`, hoặc dự án cấp thêm ngân sách duyệt trong guideline. Không hạ `conf` chỉ để hàng đợi vừa 80 ảnh, vì như thế là Release giả (mục 3).

**(b) Ghi chú AI** (dùng Claude Code):
- **Prompt chính:** đưa docstring của từng stub (`label_format`, `gate`, `autofix`, `second_opinion`) + file test tương ứng, yêu cầu viết đúng đầu vào/đầu ra trong docstring, lấy ngưỡng từ `cfg`, không đọc `review_reference`/`MANIFEST`, chỉ sửa trong `ds`.
