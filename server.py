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
    """Generate a meaningful sentence using the word in daily conversation."""
    # Adjust the prompt to request a complete sentence
    prompt = f"Generate a grammatically correct and meaningful sentence using the word '{word}' in a natural, everyday conversation. For example: 'I like to {word} in the morning.'"

    try:
        result = models[model_name](
            prompt,
            max_new_tokens=20,
            num_return_sequences=1,  # Only return one sentence
            do_sample=True,
            temperature=0.9,
            top_p=0.95
        )
        sentence = result[0]["generated_text"].strip()

        # Ensure the sentence is meaningful (not too short or incomplete)
        if len(sentence.split()) < 3:  # If the sentence is too short or incomplete
            return ""  # Return an empty string if it doesn't make sense

        # Trim the sentence to 6 words to keep it short but meaningful
        sentence_words = sentence.split()
        sentence = ' '.join(sentence_words[:6])

        # Ensure the sentence contains the provided word
        if word.lower() in sentence.lower():
            return sentence
        return ""  # Return empty if the word is not found in the sentence

    except Exception as e:
        print(f"⚠️ Model {model_name} Error: {e}")
        return ""  # Return empty if there's an error in the model


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """ เพิ่ม route สำหรับ / เพื่อทดสอบว่าเซิร์ฟเวอร์ทำงานได้ """
    return {"message": "Server is live!"}

@app.get("/generate_sentence/")  # Endpoint for generating a sentence
async def generate_sentence(word: str):
    """Generate a meaningful sentence from the provided word"""
    try:
        # List to collect results from each model
        results = []
        
        # Process each model one at a time
        for model in models:
            result = await generate_from_model(model, word)
            if result:  # If the model returns a valid result
                results.append(result)
        
        # Return the first valid result
        if results:
            return {"sentence": results[0]}  # Return the first valid sentence from the model
        else:
            # If no valid sentence is generated, use the fallback sentence
            fallback_sentence = f"Here is a fallback sentence with the word '{word}': 'I like {word} in the morning.'"
            fallback_words = fallback_sentence.split()
            fallback_sentence = ' '.join(fallback_words[:6])  # Limit to 6 words
            return {"sentence": fallback_sentence}

    except asyncio.TimeoutError:
        return {"error": "⏳ Some models took too long to respond."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use the port set by Render
    uvicorn.run(app, host="0.0.0.0", port=port)  # Use 0.0.0.0 to allow external access
