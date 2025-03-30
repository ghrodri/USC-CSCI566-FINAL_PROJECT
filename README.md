# 🎯 Multimodal Sentiment Recognition in Political Speeches

This repository contains the code and models for our CSCI 566 project: **Multimodal Sentiment Recognition in Political Speeches Using Utterance-Level Segmentation**.

## 📘 Overview

We propose a multimodal sentiment classification system that leverages **utterance-level segmentation** (via Whisper) to align text, audio, and video features in political speech videos. The system performs **sentiment classification** into **positive**, **neutral**, or **negative** at the utterance level.

### Key Contributions
- Trained **unimodal models** (text, audio, and video) as independent baselines.
- Explored **early fusion strategies** using MLP and attention mechanisms.
- Currently expanding to **late fusion** using decision-level integration.
- Built a custom dataset of political speeches with timestamped utterances.

## 🧠 Modalities

| Modality | Model Used | Features |
|----------|-------------|----------|
| Text     | RoBERTa / BERT | [CLS] token embeddings |
| Audio    | MLP on handcrafted features | MFCCs, Chroma, Spectral Contrast |
| Video    | (In progress) BiLSTM + Attention | Facial Action Units via OpenFace |

## 📊 Performance Summary

| Model                    | Modality | Accuracy | Weighted F1 | Remarks                                         |
|-------------------------|----------|----------|-------------|-------------------------------------------------|
| BERT                    | Text     | ~66%     | ~65%        | Unimodal text classifier                        |
| RoBERTa Base            | Text     | ~73%     | ~72%        | Slightly outperforms BERT                      |
| RoBERTa Large           | Text     | ~72%     | ~71%        | Robust but similar to Base                     |
| Audio (Classification)  | Audio    | ~67%     | ~66%        | Unimodal audio classification                  |
| Audio (Intensity)       | Audio    | ~69%     | ~68%        | Evaluates emotional intensity                  |
| Early Fusion (MLP)      | Fusion   | ~45.5%   | ~30%        | Early fusion using a multilayer perceptron     |
| Early Fusion (Attention)| Fusion   | ~66.5%   | ~64%        | Early fusion using an attention mechanism      |
