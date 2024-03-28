import os
import torch
from datasets import load_dataset
from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
import torch
print(torch.__version__)
print(torch.cuda.is_available())
import bitsandbytes as bnb
print(bnb.__version__)

from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
import pandas as pd



output_dir = "./saved_model"

model = AutoModelForCausalLM.from_pretrained(output_dir)
model.config.output_hidden_states = True
model.config.do_sample = False #True하면 랜덤 샘플로 다음 문장 생성(정확도는 낮지만 다양성 가능) False는 가장 높은 확률 생성
model.config.use_cache = False #모델이 텍스트 생성할때 중간결과 캐싱 X(긴 텍스트에 경우 성능에 지장이 있을수도..)
model.config.pretraining_tp = 1 #tensor_parallelism(텐서 병렬화) 대규모 모델을 여러 gpu에 걸쳐서 효과적으로 학습하는 기법, 1:사용X
model.config.temperature = 1.0
model.config.top_p = 1.0


tokenizer = AutoTokenizer.from_pretrained(output_dir)

# 로깅 수준 설정

logging.set_verbosity(logging.CRITICAL)

df = pd.read_csv("")

text = "How many people does the Smart Finance Department recruit per year?"
text_inputs = tokenizer(text, return_tensors="pt") #pt=pytorch-tensor, 이외에도 np, tf가능
outputs = model(**text_inputs)

hidden_states = outputs.hidden_states
last_layer_features = hidden_states[-1]
print(last_layer_features.shape)

hidden_states_np = last_layer_features.cpu().detach().numpy()


file_path = "C:\\Users\\DESKTOP\\Documents\\연구\\MultiModal\\feature_map\\hidden_states.npy"

np.save(file_path, hidden_states_np)


prompt = "1.RI: F-18-FDG: 8.86 MCI 2.bst 104mg/dl 3. INJ Site: LEFT ELBOW Check: Inject F-18 FDG and get a fully body video from skull base to INGUINAL Area after 1 hour.Before the FDG injection, duffataline 135mg was taken oral, and lasix 10mg was intravenously after the FDG injection.Clinical findings: Cancer Breast W/U Reading finding -Whole Body Pet: HM in Outer Pericenter with MaxSuv 10.9 Multicentric Breast Metabolism in Inner Pericentral of Right Breast with MaxSUV 4.7 Additi Onal Metabolic LESION in Lower Pericentral of Right Breast with MaxSuv 1.9.Multiple Metastatic LNS in Right Axilla Level II with Maxsuv 10.6 MILD MELD METABOLISM in left adnexa.Abormal hyperemtabolic lesion is not observed in other images."
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1024)
result = pipe(f"<s>[INST] {prompt} [/INST]")
for i, output in enumerate(result):
    print(f"Result {i+1}:")
    print(output['generated_text'])