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

@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """ เพิ่ม route สำหรับ / เพื่อทดสอบว่าเซิร์ฟเวอร์ทำงานได้ """
    return {"message": "Server is live!"}


@app.get("/generate_sentence/")
async def generate_sentence(word: str):
    """สร้างประโยคจากคำที่ให้มาในชีวิตประจำวัน"""
    try:
        # สร้าง list สำหรับเก็บผลลัพธ์จากแต่ละโมเดล
        results = []
        
        # ทำงานทีละโมเดล
        for model in models:
            result = await generate_from_model(model, word)
            if result:  # ถ้าโมเดลให้ผลลัพธ์ที่ใช้ได้
                results.append(result)
        
        # ตรวจสอบผลลัพธ์ที่ไม่ว่าง
        if results:
            return {"sentence": results[0]}  # ส่งผลลัพธ์จากโมเดลแรกที่ใช้ได้
        else:
            # ถ้าไม่สามารถสร้างประโยคได้จากโมเดล จะใช้ประโยคที่ตั้งไว้
            fallback_sentence = f"Here is a fallback sentence with the word '{word}': 'I like {word}.'"
            fallback_words = fallback_sentence.split()
            fallback_sentence = ' '.join(fallback_words[:6])
            return {"sentence": fallback_sentence}

    except asyncio.TimeoutError:
        return {"error": "⏳ Some models took too long to respond."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # ใช้พอร์ตที่ Render กำหนดให้
    uvicorn.run(app, host="0.0.0.0", port=port)  # ใช้ 0.0.0.0 เพื่อให้เครื่องอื่นสามารถเข้าถึงได้
