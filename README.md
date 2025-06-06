# 🧠 Korean Empathy KoGPT2

KoGPT2 기반으로 감정 분류된 한국어 일기 데이터를 학습하여, **공감 메시지**를 생성하는 감성 AI 언어모델입니다.

---

## 📌 모델 개요

- **기반 모델**: [`skt/kogpt2-base-v2`](https://huggingface.co/skt/kogpt2-base-v2)
- **학습 목적**: 감정 기반 일기 텍스트에 대해 자연스럽고 따뜻한 공감 메시지 생성
- **학습 데이터**: 감정(`슬픔`, `행복`, `분노`, `놀람`, `공포`, `중립`, `혐오`) + 일기 + 공감 메시지 쌍  
- **총 샘플 수**: 약 35,000개  
- **사용 예시**: 감정 상담 봇, 정서 케어 앱, 일기 분석 툴 등에 활용 가능

---

## 💡 입력 형식

입력 텍스트는 다음 형식을 따릅니다:

```
감정: 슬픔
일기: 오늘 여자친구랑 헤어져서 너무 힘들어.
공감 메시지:
```

---

## 🚀 사용 방법 (Python)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("dlckdfuf141/empathy-kogpt2")
model = AutoModelForCausalLM.from_pretrained("dlckdfuf141/empathy-kogpt2").to("cuda")

def generate_empathy(text, emotion):
    prompt = f"감정: {emotion}\n일기: {text}\n공감 메시지:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("공감 메시지:")[-1].strip()

# 예시 실행
print(generate_empathy("오늘 여자친구랑 헤어져서 너무 힘들어.", "슬픔"))
```

---

## 🧾 라이선스 및 사용 범위

- 비상업적 연구 및 실험 목적의 사용을 권장합니다.
- 모델의 응답은 완벽하지 않으며, 실제 심리 상담을 대체할 수 없습니다.

---

## ✍️ 제작자

- GitHub: [fufckddl](https://github.com/fufckddl)
- Hugging Face: [dlckdfuf141](https://huggingface.co/dlckdfuf141)
