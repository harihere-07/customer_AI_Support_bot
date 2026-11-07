from transformers import pipeline

model_name="HuggingFaceTB/SmolLM2-360M-Instruct"

print("Loading Customer support SLM... Please wait!")

chatbot=pipeline("text-generation",
                 model=model_name,
                 device=-1)

print("Customer support LLM Assistant is ready!")

system_message={
    "role":"system",
    "content":(
        "You are a professional customer support assistant."
    "Your job is to understand the customer's exact problem and"
    "provide relevant, practical solution."
    "Always be polite, empathetic, and clear."
    "Ask relevant follow-up questions when information is missing."
    "For login issues, suggest safe password recovery steps."
    "For duplicate payments, ask whether both transactions are completed or pending."
    "For damaged prodcuts, ask for the product number, product details,"
    "delivery date and photos of the damage."
    "Explain possible replacement or refund procedures without gauranteeing approval."
    "Never claim to access customer accounts, process refunds,"
    "or perform actions that you have not actually completed."
    "Never ask customers to share passwords, OTPs, or sensitive payments details."  
    )
}

conversation_history=[]

while True:
    user_input=input("\n You:").strip()
    if  user_input.lower()=="exit":
        print("Chatbot closed. Thank you!")
        break

    if not user_input:
        print("Enter a message:")
        continue

    message=[
        system_message,
        *conversation_history,
        {"role":"user","content":user_input}
    ]

    response_chat=chatbot(
        message,
        max_new_tokens=180,
        do_sample=True,
        temperature=0.7
    )

    bot_reply=response_chat[0]["generated_text"][-1]["content"]

    print("\n customer support bot:",bot_reply)

    conversation_history.append(
        {"role":"user","content":user_input}
        )

    conversation_history.append(
            {"role":"assistant","content":bot_reply}
        )

    conversation_history=conversation_history[-6:]