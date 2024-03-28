# MMLseg: MultiModal Liver Tumor Segmentation with CT and Report Data

MMLseg 프로젝트는 CT 이미지와 보고서 데이터를 활용한 다중 모달 리버 튜머 세그멘테이션을 목표로 합니다.

## 목차

1. [LLM](#1-llm)
2. [Model](#2-model)
3. [Preprocess](#3-preprocess)
4. [Train & Inference](#4-train--inference)

---

## 1. LLM

- **finetuning.py**: `LLAMA-2-7b-hf` 모델을 로컬에서 파인튜닝합니다.
- **llm-inference.py**: 파인튜닝된 모델을 사용하여 텍스트 임베딩을 수행합니다.

## 2. Model

- **swin_unetr.py**: DownScale layer를 추가하고 Skip Connection에 Interactive Alignment 모듈을 적용합니다.
- **swin_unetr_ia_on_all_encoder.py**: DownScale layer를 추가하고 Encoder output에 Interactive Alignment 모듈을 적용합니다.

## 3. Preprocess

- **process.py**: Liver을 최대한 포함하도록 320x320 사이즈로 이미지를 크롭합니다.

## 4. Train & Inference

- **mymodel.py**: 최종적으로 모델을 training합니다.
- **inference.py**: `.pth` 파일을 로드하여 추론을 수행합니다.
