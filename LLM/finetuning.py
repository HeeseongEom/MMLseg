import torch
print(torch.__version__)
print(torch.cuda.is_available())
import bitsandbytes as bnb
print(bnb.__version__)

from transformers import(
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig)


import huggingface_hub
huggingface_hub.login()


dataPath = "C:\\Users\\DESKTOP\\Documents\\연구\\MultiModal\\dataset\\"
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

###Flash Attention 메커니즘 사용하면 어텐션 훨씬 빨라짐 --> 대신 torch가 업데이트해야하는듯



base_model = "NousResearch/Llama-2-7b-hf"

rscode_dataset = "heeseongE/customllm"
# guanaco_dataset = "mlabonne/guanaco-llama2-1k“

# Fine-tuned model
new_model = "llama-2-7b-heeseongcustom"
dataset = load_dataset(rscode_dataset, split="train")

print(dataset[0])
print(len(dataset))
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained(base_model)
print(config.do_sample)
print(config.temperature)  # 생성 시 다양성을 조절하는 온도 파라미터
print(config.top_p)  # 생성 시 사용되는 토큰 확률의 누적 비율 제한

config.do_sample = True  # 샘플링 사용
config.temperature = 0.9  # 생성 시 다양성을 조절하는 온도 파라미터
config.top_p = 0.6  # 생성 시 사용되는 토큰 확률의 누적 비율 제한
config.use_cache = False  # 캐시 사용 비활성화
config.output_hidden_states = True  # 특성 추출 활성화


model = AutoModelForCausalLM.from_pretrained(
    base_model, #base_model:해당 구조에 맞게 weight를 불러옴, quantization~:앞선 정의대로 양자화, device_map:""는 모든 문자열, 첫번째 gpu 사용
    config=config,
    quantization_config=quant_config,
    device_map={"": 0}
    #output_hidden_states=True #feature extraction할거면 True
)



'''generation_config = GenerationConfig(do_sample = True, #True하면 랜덤 샘플로 다음 문장 생성(정확도는 낮지만 다양성 가능) False는 가장 높은 확률 생성
temperature = 1.0,
top_p = 1.0).to_dict() #모델이 텍스트 생성할때 중간결과 캐싱 X(긴 텍스트에 경우 성능에 지장이 있을수도..)
#model.config.pretraining_tp = 1 #tensor_parallelism(텐서 병렬화) 대규모 모델을 여러 gpu에 걸쳐서 효과적으로 학습하는 기법, 1:사용X

model.config.update(generation_config)'''
#model.config.generation = generation_config





#사전 훈련 모델형식에 맞는 tokenizer 로드(trust_remote_code를 True해야 원격으로 사용하며 변경가능한듯)
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
#EOS(문장의 끝)token을 pad_token으로 설정: pad_token은 입력데이터 길이를 맞출때 사용(eos로 길이 맞춤)
tokenizer.pad_token = tokenizer.eos_token
#입력 시퀀스의 오른쪽끝에 패딩을 추가한다는 의미
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16, # 확장될 parameter의 양
    lora_dropout=0.1, #
    r=64, #rank(내부차원) 높을수록 계산량증가, 표현력향상
    bias="none",
    task_type="CAUSAL_LM", #LoRA 작업유형 : causal_lm->우리 pretrained 유형임(텍스트 생성에 사용됨)
)

training_params = TrainingArguments(
    output_dir="./results", #여기에 log, checkpoint 등등 저장
    num_train_epochs=5, # 모델 훈련 총 epoch
    per_device_train_batch_size=1, # 각 gpu당 batch
    gradient_accumulation_steps=1, # 1만큼 gradient를 누적한후 업데이트->큰 배치사이즈를 시뮬레이션하여 매모리 사용량줄임
    optim="paged_adamw_32bit", #optimizer
    save_steps=25, #25마다 체크포인트 저장
    logging_steps=25, #25마다 로그 출력
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False, #16비트 부동소수점 사용여부
    bf16=False,
    max_grad_norm=0.3, #gradient clipping할 norm값:0.3
    max_steps=-1, #실행할 최대 스텝수(-1은 제한없음 의미)
    warmup_ratio=0.03, #학습초기에 점진적으로 학습률 증가시킴
    group_by_length=True, #유사한 길이의 예제를 함께 배치하도록 함 -> 훈련효율성 증가
    lr_scheduler_type="constant", #학습 스케줄러 유형: constant는 학습률 유지
    #report_to="tensorboard" #훈련로그를 tensorboard에 보고
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_params, #파인튜닝용 LoRA등의 추가설정
    dataset_text_field="text", #데이터셋 내의 텍스트 필드의 이름 지정
    max_seq_length=None, #입력 무제한
    tokenizer=tokenizer, #토크나이저 지정
    args=training_params,
    packing=False, #입력데이터 패킹(시퀀스 효율적으로 배치할건지) 여부
)

#output_dir = "./saved_model"


###FineTuning###
trainer.train()


###Inference###
logging.set_verbosity(logging.CRITICAL)

prompt = "1.RI: F-18-FDG: 8.86 MCI 2.bst 104mg/dl 3. INJ Site: LEFT ELBOW Check: Inject F-18 FDG and get a fully body video from skull base to INGUINAL Area after 1 hour.Before the FDG injection, duffataline 135mg was taken oral, and lasix 10mg was intravenously after the FDG injection.Clinical findings: Cancer Breast W/U Reading finding -Whole Body Pet: HM in Outer Pericenter with MaxSuv 10.9 Multicentric Breast Metabolism in Inner Pericentral of Right Breast with MaxSUV 4.7 Additi Onal Metabolic LESION in Lower Pericentral of Right Breast with MaxSuv 1.9.Multiple Metastatic LNS in Right Axilla Level II with Maxsuv 10.6 MILD MELD METABOLISM in LeIADNEXA.Abormal hyperemtabolic lesion is not observed in other images."
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=1000, num_beams=2)
result = pipe(f"<s>[INST] {prompt} [/INST]")
#print(result[0]['generated_text'])

for i, output in enumerate(result):
    print(f"Result {i+1}:")
    print(output['generated_text'])

text = "How many people does the Smart Finance Department recruit per year?"
text_inputs = tokenizer(text, return_tensors="pt") #pt=pytorch-tensor, 이외에도 np, tf가능
outputs = model(**text_inputs)

hidden_states = outputs.hidden_states
last_layer_features = hidden_states[-1]
print(last_layer_features.shape)


output_dir = "./saved_model"

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
trainer.save_model(output_dir)