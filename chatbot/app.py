from gpt4all import GPT4All

model = GPT4All("ggml-mistral-7b-instruct.bin")
with model.chat_session():
    response = model.generate("Halo, apa kabar?", max_tokens=200)
    print(response)
