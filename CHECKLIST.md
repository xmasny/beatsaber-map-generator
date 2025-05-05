# 📝 Beat Saber Note Prediction — Training Checklist

## ✅ Phase 1: Onset Prediction (In Progress)

- [x] Define onset label generation (`onsets_array` from notes)
- [x] Create dataset with `audio`, `onset`, `beats` (optional), and `condition`
- [x] Model architecture: `SimpleOnsets`
- [x] Use sigmoid output + `BCEWithLogitsLoss`
- [x] Start training
- [x] Track metrics: precision, recall, F1
- [x] Save model checkpoint

---

## 🔄 Phase 2: Note Prediction (Upcoming)

### 🧾 Data Preparation (Pending)

- [x] Load full mel spectrogram
- [ ] For each note: compute `frame = int(round(time / FRAME))`
- [ ] Stack ±3 context frames → input shape: `(T, 7 * n_mels)`
- [ ] Encode label: `label = encode_label(...) + 1` ∈ `[1–216]`
- [ ] Create sparse `(T,)` label array (only note frames included)
- [ ] Save `note_labels` into `.npz`

### 🧠 Model Setup (Ready)

- [ ] Architecture: `AudioSymbolicNoteSelector`
- [ ] Input: stacked mel, optional symbolic input (e.g., onsets)
- [ ] Output: `Linear(..., 217)` (class 0 = “no note”)
- [ ] Loss: `CrossEntropyLoss`
- [ ] Train on sparse frames for now (note-only)
- [ ] Validate only on note predictions

---

## 🚀 Phase 3: Framewise Note Training (Future)

- [ ] Expand labels to full `(T,)` arrays with class `0` for “no note”
- [ ] Train on every frame (framewise classification)
- [ ] Add class weights to down-weight `0`
- [ ] Evaluate full sequence accuracy and false positive rate

---

## 🧪 Phase 4: Advanced Experiments (Optional)

### 🎯 Multi-Head Attribute Prediction (color, direction, x, y)

- [ ] Store separate per-attribute labels in `.npz`
- [ ] Define model with multi-output heads
- [ ] Output:
  - color (2), direction (9), x (4), y (3)
- [ ] Use 4-way `CrossEntropyLoss`
- [ ] Evaluate per-attribute accuracy and confusion
- [ ] Compare against joint label model (217-class)

### 🧠 Use Cases

- [ ] Easier debugging of mistakes (e.g., right color, wrong direction)
- [ ] Can train on partial labels
- [ ] Can combine with joint label model in multi-task setup

## 🔊 Phase 5: Use Onset Model in Note Training (Optional)

- [ ] Pass onset scores as `symbolic` input to note model
- [ ] Freeze onset model weights during note training
- [ ] Use `sigmoid(onset_logits)` as guide
- [ ] Evaluate if this improves accuracy or reduces false positives
- [ ] Combine onset + note into one multitask model

## Optional ideas

- Models with more than 1 note per frame (e.g., 2 notes in 1 frame) (onsets and notes)
- train bomb notes and obstacles (onsets and notes)