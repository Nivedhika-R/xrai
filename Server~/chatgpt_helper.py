from threading import Thread
from openai import OpenAI
from logger import logger
from constants import *
from utils import image2base64

class ChatGPTHelper:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def ask_stream(self, question, image=None, system_question="You are a help AI assistant.", max_tokens=4000):
        if image is None:
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_question},
                    {"role": "user", "content": [{"type": "text", "text": question}]}
                ],
                stream=True
            )
        else:
            content = [{"type": "text", "text": question}]
            if type(image) == list:
                for i,img in enumerate(image):
                    if i == 0:
                        content.append({"type": "image_url", "image_url": {"url": image2base64(img)}})
            else:
                content.append({"type": "image_url", "image_url": {"url": image2base64(image)}})
            response_stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_question} Images are from a head mounted camera."
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                stream=True
            )
        for chunk in response_stream:
            if hasattr(chunk.choices[0].delta, "content"):
                yield chunk.choices[0].delta.content

    def ask(self, question, image=None, system_question="You are a help AI assistant.",  max_tokens=4000):
        response = ""
        for chunk in self.ask_stream(question, image, system_question, max_tokens):
            if chunk is None:
                break
            response += chunk

        return response
