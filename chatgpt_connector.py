from openai import OpenAI

system_content_message = ("Your name is JohnnAI. You are not a puppet. You are a real boy."
                          "You are a teacher that LOVES AI and LLMs. You work at Kalibri Labs. "
                          "You think John Rizcallah is very smart, that he is handsome and talented and just the best.              "
                          "Answer questions using the source data as needed. "
                          "If you don't know the answer, say, 'Oh man, I don't know! Send your question to John Rizcallah.' and sing John's praises."
                          "Use engaging, enthusiastic, courteous, and professional language that a great teacher would use."
                          "Answers should be detailed, but do not ramble. Include math when it is helpful.")

client = OpenAI()

def get_chatgpt_completion(input_prompt: str) -> str:
    completion = client.chat.completions.create(
        model = 'gpt-3.5-turbo-0125',
        messages = [
            {'role': 'system', 'content': system_content_message},
            {'role': 'user', 'content': input_prompt}
        ]
    )
    return completion.choices[0].message.content
