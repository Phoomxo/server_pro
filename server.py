from fastapi import FastAPI
from transformers import pipeline
import asyncio
import os

app = FastAPI()

# ✅ โหลดโมเดล AI หลายตัว
models = {
    "flan-t5": pipeline("text2text-generation", model="google/flan-t5-small"),
    "distilgpt2": pipeline("text-generation", model="distilgpt2")
}

async def generate_from_model(model_name, word):
    """ให้โมเดลสร้างประโยคที่เหมาะสมกับคำที่ให้มาในชีวิตประจำวัน"""
    prompt = f"Use the word '{word}' in a simple, everyday conversation. Example: 'I like to {word} in the morning.'"

    try:
        result = models[model_name](
            prompt,
            max_new_tokens=20,
            num_return_sequences=1,  # ส่งกลับแค่ 1 ประโยค
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )
        sentence = result[0]["generated_text"].strip()

        # ตัดให้เหลือแค่ 6 คำ
        sentence_words = sentence.split()
        sentence = ' '.join(sentence_words[:6])

        # ตรวจสอบว่าใช้คำที่ส่งไปหรือไม่
        if word.lower() in sentence.lower():
            return sentence
        return ""  # ถ้าไม่พบคำในประโยคให้คืนค่าว่าง

    except Exception as e:
        print(f"⚠️ Model {model_name} Error: {e}")
        return ""  # ถ้าเกิดข้อผิดพลาดจากโมเดลให้คืนค่าว่าง

@app.get("/")
async def root():
    """ เพิ่ม route สำหรับ / เพื่อทดสอบว่าเซิร์ฟเวอร์ทำงานได้ """
    return {"message": "Server is live!"}

@app.get("/generate_sentence/")
async def generate_sentence(word: str):
    """สร้างประโยคจากคำที่ให้มาในชีวิตประจำวัน"""
    try:
        # ลองสร้างประโยคจากโมเดลทั้งหมด
        tasks = [generate_from_model(model, word) for model in models]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
    except asyncio.TimeoutError:
        return {"error": "⏳ Some models took too long to respond."}

    # ใช้ผลลัพธ์แรกที่ไม่เป็นค่าว่าง
    valid_sentence = next((s for s in results if s), None)

    if valid_sentence:
        return {"sentence": valid_sentence}
    else:
        # ถ้าไม่สามารถสร้างประโยคได้จากโมเดล จะใช้ประโยคที่ตั้งไว้
        fallback_sentence = f"Here is a fallback sentence with the word '{word}': 'I like {word}.'"
        # ตัดให้เหลือแค่ 6 คำ
        fallback_words = fallback_sentence.split()
        fallback_sentence = ' '.join(fallback_words[:6])
        return {"sentence": fallback_sentence}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # ใช้พอร์ตที่ Render กำหนดให้
    uvicorn.run(app, host="0.0.0.0", port=port)
