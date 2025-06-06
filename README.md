# 🧠 한국어 공감 메시지 생성 모델 (KoGPT2 기반)

**KoGPT2** 기반의 공감 메시지 생성 모델입니다.  
사용자가 작성한 일기와 감정 정보를 바탕으로, 상황에 어울리는 따뜻한 공감 문장을 생성합니다.

---

## ✅ 예시

**입력**
감정: 슬픔
일기: 오늘 여자친구랑 헤어져서 너무 힘들어.
공감 메시지:
**출력**
마음이 많이 힘들었겠네. 괜찮아, 다 지나갈 거야.
---

## 📌 모델 정보

- **기반 모델**: `skt/kogpt2-base-v2`
- **학습 데이터**: 감정별 5,000개씩 총 35,000개
- **지원 감정**:
  - 슬픔
  - 행복
  - 분노
  - 공포
  - 놀람
  - 중립
  - 혐오

- **입력 형식**
감정: [감정]
일기: [사용자 작성 문장]
공감 메시지:

- **출력 형식**
상황에 어울리는 한 문장 공감 메시지


---

## 💻 사용 방법 (Python 예시)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("fufckddl/KoreanEmpathyModel")
model = AutoModelForCausalLM.from_pretrained("fufckddl/KoreanEmpathyModel").to("cuda")

def generate_empathy(text, emotion):
  prompt = f"감정: {emotion}\n일기: {text}\n공감 메시지:"
  inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
  output = model.generate(
      **inputs,
      max_new_tokens=60,
      do_sample=True,
      top_p=0.95,
      temperature=0.8,
      pad_token_id=tokenizer.pad_token_id,
      eos_token_id=tokenizer.eos_token_id
  )
  response = tokenizer.decode(output[0], skip_special_tokens=True)
  return response.split("공감 메시지:")[-1].strip()

# 사용 예시
print(generate_empathy("오늘 너무 지치고 힘들었어.", "슬픔"))```

🛠️ 학습 환경
GPU: NVIDIA RTX 4060

학습 시간: 약 2시간

배치 크기: 2

에폭 수: 3

최대 토큰 길이: 128

