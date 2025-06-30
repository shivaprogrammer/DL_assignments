# Deep Learning 

> **Course:** CSL 7590
> **Author:** Shivani Tiwari (M24CSA029)
> **Assignments Covered:** 1 – 4

---

## Table of Contents

1. [Purpose](#purpose)
2. [Global Comparison](#global-comparison)
3. [Assignment 1 – MNIST NN from Scratch](#assignment-1)
4. [Assignment 2 – Multi‑Task CNN on CIFAR‑100](#assignment-2)
5. [Assignment 3 – Sketch‑RNN on QuickDraw](#assignment-3)
6. [Assignment 4 – Data‑Free Adversarial KD](#assignment-4)

---

## Purpose

This repository bundles **four progressive deep‑learning assignments**. Each task highlights a different facet of modern DL—hand‑coded neural nets, hierarchical vision, sequence modelling, and data‑free knowledge distillation—giving a holistic view of techniques and best practices.

---

## Global Comparison

|     # | Topic & Dataset          | Core Architecture                   | Key Metric(s)                                  | Highlight                         |
| ----: | ------------------------ | ----------------------------------- | ---------------------------------------------- | --------------------------------- |
| **1** | MNIST digits (70 K imgs) | 2‑hidden‑layer FC (scratch)         | 94.7 % test acc (mini‑batch, 90 : 10 split)    | Manual BP + GD variants           |
| **2** | CIFAR‑100 (60 K imgs)    | Shared CNN + 3 heads                | 62.5 % group acc (90 : 10, severity loss)      | Severity‑weighted multi‑task loss |
| **3** | QuickDraw (5 classes)    | Bi‑LSTM encoder‑decoder + attention | 0.0029 loss / 93.7 % draw acc (10 epochs)      | Real‑time stroke generation       |
| **4** | CIFAR‑100 (data‑free)    | ResNet‑34 T / 2 students / DCGAN G  | 8 % acc (Student 2, 10 % split) — 4.2 M params | GAN‑driven distillation           |

---

## Assignment 1

### Description & Goal

Build a **fully‑connected neural network from scratch** (Python/NumPy—no DL libs) to classify MNIST digits, while experimenting with weight‑initialisation schemes, three GD variants, L2 regularisation and three train‑test splits.

### System Architecture

```
784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
```

*Weights initialised with Random / Xavier / He; manual forward & backward passes.*

### Data Flow

1. **Load & Normalise** MNIST → \[0, 1].
2. **Split** 70 : 30 / 80 : 20 / 90 : 10.
3. **Train Loop** — select GD variant, run 25 epochs, store metrics.
4. **Evaluate** — confusion matrix + plots.

### Results & Comparison

|   Split | GD Type    |  Train Acc |   Test Acc | Loss Trend  |   |
| ------: | ---------- | ---------: | ---------: | ----------- | - |
| 70 : 30 | Batch      |     11.3 % |     11.0 % | flat        |   |
| 70 : 30 | SGD        |     89.8 % |     89.1 % | noisy ↓     |   |
| 70 : 30 | Mini‑Batch | **94.5 %** | **93.9 %** | smooth ↓    |   |
| 90 : 10 | Mini‑Batch | **94.7 %** | **94.2 %** | smoothest ↓ |   |

**Insights** – Mini‑batch GD consistently dominates; larger train splits boost generalisation; He + ReLU combo converges fastest.

---

## Assignment 2

### Description & Goal

Design a **single CNN backbone** with **three output heads** to predict fine (100), superclass (20) and custom group (9) labels for CIFAR‑100. Introduce a severity‑weighted loss that penalises cross‑group mistakes more harshly.

### System Architecture

* **Feature Extractor:** 3 × \[Conv‑BN‑ReLU] → MaxPool → Dropout.
* **Heads:** three parallel FC stacks for fine / super / group logits.
* **Loss:** Cross‑entropy modulated by a **severity matrix** (same superclass < same group < diff group).

### Data Flow

1. **Custom Dataset** maps fine labels to superclass & group.
2. **Weighted Sampler** mitigates class imbalance.
3. **Train** 20 epochs under 70:30 / 80:20 / 90:10 splits.
4. **Log** per‑head accuracies + confusion matrices.

### Results & Comparison

|   Split | Final Loss |    Fine Acc |   Super Acc |   Group Acc |   |
| ------: | ---------: | ----------: | ----------: | ----------: | - |
| 70 : 30 |      0.492 |     52.06 % |     52.55 % |     54.95 % |   |
| 80 : 20 |      0.483 |     53.56 % |     53.79 % |     57.44 % |   |
| 90 : 10 |  **0.459** | **55.65 %** | **56.01 %** | **59.25 %** |   |

*With severity loss:* group accuracy rises to **62.48 %** on 90 : 10, confirming the penalty’s utility.

---

## Assignment 3

### Description & Goal

Implement a **SketchRNN‑style seq‑to‑seq model** that, given a class label, sequentially draws a sketch (dx, dy, pen‑state) for five symbols.

### System Architecture

* **Encoder:** class‑embedding → 2‑layer Bi‑LSTM.
* **Attention:** additive, computed each decoder step.
* **Decoder:** 2‑layer LSTM outputs Δx, Δy & pen logits.

### Data Flow

1. **Download & cache** NDJSON strokes.
2. **Pre‑process:** normalise, pad to 300, label‑encode.
3. **Train** AdamW 10 epochs (batch 32).
4. **Live visualiser** animates strokes.

### Results

Loss dives from **0.008 → 0.0029**, stabilising after epoch 3; draw accuracy plateaus at **≈ 93.7 %**.

---

## Assignment 4

### Description & Goal

Apply **Data‑Free Adversarial Knowledge Distillation**: train two lightweight students (≈ 10 % & 20 % params of ResNet‑34 teacher) using a GAN‑like generator that fabricates training images—no real CIFAR‑100 data.

### System Architecture

```
Noise → DCGAN‑G → synthetic imgs → Teacher (T) + Student (S)
                     ↑                      |
                     └────── adversarial loss┘
```

*Adversarial loop alternates Student and Generator updates with MAE‑based objectives.*

### Data Flow

1. **Setup:** load teacher (85.1 % acc), build students & generator.
2. **Loop:** ks = 5 student steps, kg = 1 generator step per epoch.
3. **Checkpoint** best accuracy every 5–10 epochs; save models & sample images.
4. **Final eval** on 10 % & 20 % held‑out test subsets.

### Results & Comparison

|     Model |        Params |  10 % Test |  20 % Test |   |
| --------: | ------------: | ---------: | ---------: | - |
| Student 1 | 2.16 M (10 %) |     5.70 % |     4.90 % |   |
| Student 2 | 4.26 M (20 %) | **8.00 %** | **6.00 %** |   |

**Takeaways** – Low‑resolution (32 × 32) images ease training; bigger student capacity helps; long runs (≥ 10 k epochs) and balanced S/G updates curb mode collapse.


