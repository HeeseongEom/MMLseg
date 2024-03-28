# MMLseg: MultiModal Liver Tumor Segmentation with CT and Report Data

<br><br>

#LLM
<br><br>
finetuning.py: LLAMA-2-7b-hf 모델을 로컬에서 파인튜닝
llm-inference.py: 튜닝된 모델을 통해 텍스트 임베딩

###model
<br><br>
swin_unetr.py: DownScale layer 추가 & Skip Connection에 Interactive Alignment 모듈을 적용
swin_unetr_ia_on_all_encoder.py: DownScale layer 추가 & Encoder output에 Interactive Alignment 모듈을 적용

###preprocess
<br><br>
process.py: liver을 최대한 포함하는 320*320 사이즈의 크롭 진행


###train_inter
<br><br>

mymodel.py: 최종적으로 모델을 training
inference.py: .pth 로드하여 추론수행
