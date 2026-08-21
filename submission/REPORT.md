# Lab 21 — Evaluation Report

**Họ tên**: Đoàn Nhật Nam  **MSSV**: 2A202601123  **Ngày**: 2026-08-21
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `T4 16GB`

> Mọi con số dưới đây phải khớp với file trong `results/`. Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | Ticket CSKH tiếng Việt → JSON triage 4 trường (250 mẫu) |
| Train / val | 200 / 50 (seed 42) |
| `max_length` | 256 — p95 đo được là 98 *(results/token_stats.json)* |
| `MASK_MODE` | assistant-only |
| Epochs / max_steps | 30 |

**Template có giữ khối `<think>` không?** `có` — *(results/template_check.json)*
Nếu không: bạn đã xử lý thế nào?
Template `qwen-2.5` / `qwen-3.5` của model giữ nguyên khối `<think>` trong chat template (`reasoning preserved — safe to train on traces`). Do đó, luồng huấn luyện an toàn và không bị cắt cụt reasoning traces.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | true |
| Câu hỏi KHÔNG nằm trong loss | true |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3331.2 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 1056.6 |
| (c) LoRA fine-tune | 0.9375 | 0.6250 | 1.0000 | 1558.7 |

**(b) có thật sự mạnh hơn (a) không?** `có` — Baseline (b) vượt trội hoàn toàn so với (a): Target tăng từ 0.0% lên 68.75%, tỷ lệ Format JSON hợp lệ tăng từ 0.0% lên 100.0%, đồng thời Latency giảm hơn 3 lần (từ 3331.2 ms xuống 1056.6 ms) nhờ cấu trúc prompt rõ ràng giúp mô hình không sinh văn bản thừa.
Bạn có sửa `OPTIMIZED_PROMPT` không? Nếu có: **làm mạnh lên hay yếu đi**, và vì sao?
Tôi giữ nguyên bản `OPTIMIZED_PROMPT` chuẩn của labkit (khớp SHA `719e74d3b6232053`), đảm bảo tính liêm chính và khách quan khi so sánh với bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6261 | 0.9375 | 962.8 | 12.01 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 0.0001 | 0.5368 | 0.9375 | 808.1 | 12.02 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-05 | 1.5704 | 0.0000 | 936.5 | 12.01 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.8438 | 994.2 | 7.09 |

> Xếp hạng bằng cột **target**, không bằng cột train loss — chấm bằng chỉ số thay thế
> chính là Lỗi #3. Nếu hai cột cho hai thứ tự khác nhau, nói thẳng điều đó ở 4.1: đó là
> kết quả đáng giá nhất bạn đo được trong lab này.

Trả lời ba câu (mỗi câu ≥3 câu văn):

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó
thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về
*rank* so với *vị trí gắn adapter*?**
Trên tập target, `attn_only` đạt độ chính xác 0.9375 (hoà với `correct`). Tuy nhiên, trên tập huấn luyện, train loss của `attn_only` lại đạt mức 0.5368, thấp hơn đáng kể so với 0.6261 của `correct`. Thứ tự theo train loss (`attn_only` < `correct`) hoàn toàn đảo ngược so với kết quả đánh giá thực tế downstream task. Điều này chứng minh rằng việc cố tình tăng rank cực đại ($r=283$) chỉ trên các khối attention chỉ giúp mô hình ghi nhớ/overfit dữ liệu huấn luyện nhanh hơn chứ không mang lại khả năng tổng quát hóa vượt trội so với việc phân bổ adapter dàn đều trên toàn bộ các lớp `text-linear` với rank nhỏ ($r=16$). Đòn bẩy thực sự nằm ở *vị trí gắn adapter* (all-linear coverage) chứ không phải ở việc đẩy *rank* lên thật cao tại một vài vị trí cục bộ.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn
loss mà không biết LR, bạn sẽ kết luận sai điều gì?**
Cấu hình `wrong_lr` sử dụng learning rate $1\times 10^{-5}$ (thang đo của Full Fine-tuning) thay vì $1\times 10^{-4}$ (chuẩn LoRA). Kết quả là train loss hầu như không hội tụ, kết thúc ở mức rất cao 1.5704 (so với 0.6261 của `correct`) và độ chính xác target hoàn toàn bằng 0.0000. Nếu chỉ quan sát đường loss đi ngang mà không nắm được thông tin về Learning Rate, người làm rất dễ rơi vào kết luận sai lầm rằng "kỹ thuật LoRA không hiệu quả với bài toán này", "mô hình base không đủ năng lực", hoặc "tập dữ liệu bị nhiễu/thiếu hụt nghiêm trọng". Trên thực tế, adapter LoRA chỉ cập nhật một phần nhỏ tham số nên đòi hỏi mức learning rate lớn hơn gấp 5–10 lần so với full fine-tune để gradient có thể di chuyển hiệu quả trong không gian tham số rank thấp.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến
nghị "không dùng QLoRA cho dòng model này" không?**
Cấu hình `qlora` (4-bit quantization) giúp giảm mức chiếm dụng bộ nhớ VRAM đỉnh từ 12.01 GB xuống còn 7.09 GB (tiết kiệm khoảng 41% VRAM, tương đương ~4.92 GB). Tuy nhiên, cái giá phải trả là độ chính xác target bị tụt giảm nghiêm trọng từ 0.9375 xuống còn 0.8438 (mất 9.37% accuracy), thời gian huấn luyện tăng nhẹ (994.2s so với 962.8s) do overhead dequantization, và latency suy luận cũng tăng từ 1558.7 ms lên 1768.7 ms. Số đo thực nghiệm này hoàn toàn ủng hộ khuyến cáo kỹ thuật của nhà phát triển Qwen3.5: nếu tài nguyên phần cứng (như GPU T4 16GB) đã đủ khả năng chạy ở độ chính xác 16-bit (`fp16`/`bf16`), tuyệt đối không nên lạm dụng 4-bit QLoRA vì sai số lượng tử hóa sẽ làm suy giảm trực tiếp năng lực phân loại chi tiết của mô hình.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.2500` · `regression Δ = -0.1250` · `valid_trace_rate = 0.0`

Diễn giải (≥100 từ). Nếu FAILED: **vì sao**, và điều đó nói gì về bài toán của bạn?
(Một FAILED được phân tích tốt ăn điểm cao hơn một PASSED không giải thích được.)
Kết quả cổng hồi quy đạt trạng thái FAILED do chỉ số suy giảm năng lực tổng quát `regression_delta` là -0.1250 (-12.5%), vượt quá ngưỡng dung sai cho phép là 0.0200 (2.0%), mặc dù hiệu năng trên tác vụ mục tiêu tăng trưởng rất mạnh (`target_delta = +0.2500`, đưa target từ 0.6875 ở prompt b lên 0.9375 ở fine-tune). 

Nguyên nhân cốt lõi dẫn đến hiện tượng này là **Catastrophic Forgetting (Quên thảm họa)**. Trong quá trình huấn luyện 30 steps, toàn bộ 200 mẫu trong tập train đều là ticket CSKH thuần túy về phân loại khiếu nại đơn hàng. Mô hình đã điều chỉnh trọng số adapter tập trung tối đa vào cấu trúc JSON và phân loại 4 nhãn, dẫn đến việc làm xáo trộn các biểu diễn tri thức tổng quát sẵn có trong base model khi đối mặt với 15 câu hỏi thường thức ở tập `eval_regression`. 

Để khắc phục hiện tượng này trước khi đưa vào sản xuất thực tế, giải pháp bắt buộc theo khuyến nghị kỹ thuật (deck §14.3) là phải áp dụng chiến lược **Replay Buffer**: phối trộn thêm từ 1% đến 5% dữ liệu đa miền / kiến thức phổ thông (general domain instructions) vào tập huấn luyện nhằm duy trì và bảo vệ năng lực suy luận nền tảng của mô hình.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | Đúng 3/4 | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | ✅ FT thắng: Trích xuất chuẩn xác cả 4 trường, nhận diện đúng sentiment tích cực dù khách yêu cầu đổi trả gấp. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | Đúng 3/4 | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | ✅ FT thắng: Nhận diện chính xác intent hoàn tiền và thái độ tiêu cực của khách hàng. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | Đúng 4/4 | `hoan_tien`, `trung_binh`, `bình giữ nhiệt`, `tich_cuc` | ❌ **FT thua**: Model fine-tune đoán nhầm `urgency: trung_binh` (nhãn đúng là `thap` vì khách ghi "Khi nào tiện"). Mô hình bị thiên kiến gán mức độ trung bình. |
| 4 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | Đúng 4/4 | `san_pham_loi`, `trung_binh`, `nồi chiên không dầu`, `trung_tinh` | ❌ **FT thua**: Model fine-tune tiếp tục nhầm `urgency: trung_binh` do thấy cụm "Thiếu phụ kiện" lấn át từ khóa giảm nhẹ "Khi nào tiện". |
| 5 | Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi. Cảm ơn shop nhiều. | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | Đúng 3/4 | `hoan_tien`, `cao`, `đèn bàn LED`, `tich_cuc` | ✅ FT thắng: Phân loại chuẩn xác mức độ khẩn cấp `cao` khi có cụm "Quá hạn rồi". |

Có mẫu chung nào ở các ca FT thua không?
Điểm chung rõ rệt nhất ở các ca mô hình Fine-tune bị thua (đạt điểm 0.75) là lỗi dự đoán ở trường `urgency`. Mô hình có xu hướng bị thiên kiến phân phối (distribution bias) ngả về nhãn `trung_binh` khi gặp các câu có dấu hiệu khiếu nại (như "Chưa thấy tiền", "Thiếu phụ kiện"), và bỏ qua các cụm từ thể hiện sự thoải mái về mặt thời gian của người dùng như "Khi nào tiện", "Không vội", "Hỏi cho biết thôi".

---

## 7. Kết luận & điều tôi học được

**Kết luận (≥150 từ).** Bạn có nên deploy bản fine-tune này không, và vì sao? Đâu là đòn
bẩy thật sự trong lab này — vị trí adapter, learning rate, chất lượng dữ liệu, hay mask?
Dựa trên kết quả thực nghiệm toàn diện, kết luận kỹ thuật là: **Hiện tại CHƯA NÊN triển khai (deploy) bản fine-tune này trực tiếp vào hệ thống sản xuất**, mặc dù độ chính xác trên tác vụ phân loại ticket CSKH đạt mức ấn tượng 93.75% (vượt trội so với 68.75% của Prompting). Lý do then chốt là mô hình đã vi phạm cổng hồi quy an toàn (`regression_delta = -0.1250`), làm suy giảm nghiêm trọng năng lực xử lý ngôn ngữ và tri thức tổng quát. Nếu đưa vào vận hành thực tế, mô hình có nguy cơ phản hồi sai lệch nghiêm trọng đối với các tình huống ngoài phân phối (OOD) hoặc các câu hỏi tư vấn tổng quát từ khách hàng.

Phân tích sâu về các biến số thực nghiệm chỉ ra rằng: Đòn bẩy quyết định sự thành bại trong lab này chính là **Loss Masking (`assistant-only`) kết hợp với Learning Rate thang LoRA ($10^{-4}$)**. Nếu không có Loss Masking đúng, mô hình sẽ học vẹt prompt; nếu chọn sai Learning Rate (`wrong_lr`), mô hình hoàn toàn bất động. Ngược lại, việc cố gắng nâng rank cục bộ (`attn_only`) hay lượng tử hóa ép bộ nhớ (`qlora`) chỉ đem lại ảo tưởng về train loss hoặc đánh đổi trực tiếp bằng độ chính xác. Để đưa mô hình vào triển khai an toàn, bước tiếp theo bắt buộc là tái huấn luyện với 1% đến 5% tập dữ liệu replay tổng quát.

**Ba điều tôi học được** (cụ thể, không generic):
1. **Train Loss là một chỉ số đánh lừa (Deceptive Metric)**: Cấu hình `attn_only` đạt train loss thấp nhất (0.5368 so với 0.6261 của `correct`), nhưng điểm downstream target không hề vượt trội. Tuyệt đối không dùng train loss làm thước đo quyết định chất lượng mô hình.
2. **Loss Masking là điều kiện sống còn của Instruction Tuning**: Cần phải kiểm chứng bằng mã nguồn (`mask_proof.json`) giải mã ngược token để chứng minh chỉ câu trả lời được tính loss ($41.49\%$), loại bỏ $100\%$ token prompt khỏi gradient update.
3. **Prompt Engineering là mốc chuẩn (Baseline) bắt buộc phải đo trước**: Nếu không đo Baseline (b) trước, ta không thể biết liệu mức tăng điểm của Fine-tune có thực sự xứng đáng với chi phí huấn luyện và nguy cơ Catastrophic Forgetting hay không.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**
Tôi sẽ tạo một tập dữ liệu trộn (Replay Buffer) bằng cách bổ sung 3% mẫu câu hỏi hội thoại đa lĩnh vực (General Vietnamese QA) vào tập `train_seed.jsonl` và chạy lại NB3 + NB5 để kiểm chứng xem liệu mô hình có vừa đạt target $\ge 93\%$ vừa vượt qua cổng hồi quy an toàn (`regression_delta \ge -0.02`) hay không.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
