# from threading import Thread
# from openai import OpenAI
# from logger import logger
# from constants import *
# from utils import image2base64, image2base64_ollama
# from ollama import chat

# class ChatGPTHelper:
#     def __init__(self, model="gpt-4o"):
#         self.client = OpenAI(api_key=OPENAI_API_KEY)
#         self.model = model

#     def ask_stream(self, question, image=None, system_question="You are a help AI assistant.", max_tokens=4000):
#         try:
#             if image is None:

#                 response_stream = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {"role": "system", "content": system_question},
#                         {"role": "user", "content": [{"type": "text", "text": question}]}
#                     ],
#                     stream=True
#                 )
#             else:
#                 content = [{"type": "text", "text": question}]
#                 if type(image) == list:
#                     for i,img in enumerate(image):
#                         if i == 0:
#                             content.append({"type": "image_url", "image_url": {"url": image2base64(img)}})
#                 else:
#                     content.append({"type": "image_url", "image_url": {"url": image2base64(image)}})
#                 response_stream = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[
#                         {
#                             "role": "system",
#                             "content": f"{system_question} Images are from a head mounted camera."
#                         },
#                         {
#                             "role": "user",
#                             "content": content
#                         }
#                     ],
#                     stream=True
#                 )
#             for chunk in response_stream:
#                 if hasattr(chunk.choices[0].delta, "content"):
#                     yield chunk.choices[0].delta.content
#         except Exception as e:
#             logger.error(f"Error in OpenAI Call: {e}")
#             yield None
#     def ask_ollama(self, question, image=None):
#         image_bytes = []
#         for img in image:
#                 image_bytes.append(image2base64_ollama(img))

#         #print(question)
#         response = chat(
#             model='llava',
#             messages=[
#                 {
#                 'role': 'user',
#                 'content': question,
#                 'images': image_bytes,
#                 }
#             ],
#             )

#         return response.message.content

#     def ask(self, question, image=None, system_question="You are a help AI assistant.",  max_tokens=4000):
#         response = ""
#         for chunk in self.ask_stream(question, image, system_question, max_tokens):
#             if chunk is None:
#                 break
#             response += chunk

#         return response

from threading import Thread
from openai import OpenAI
from logger import logger
from constants import *
from utils import image2base64, image2base64_ollama
from ollama import chat

class ChatGPTHelper:
    def __init__(self, model="gpt-4.1"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def ask(self, question, image=None, system_question="You are a help AI assistant.", max_tokens=4000):

        if image is None:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_question},
                    {"role": "user", "content": [{"type": "text", "text": question}]}
                ],
                max_tokens=max_tokens,
                temperature=0.5,
            )
        else:
            content = [{"type": "text", "text": question}]
            if isinstance(image, list):
                for i, img in enumerate(image):
                    if i == 0:
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": image2base64(img)}
                        })
            else:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image2base64(image)}
                })
            try:
                response = self.client.chat.completions.create(
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
                    max_tokens=max_tokens,
                    temperature=0.5,
                )
            except Exception as e:
                logger.error(f"Error in OpenAI Call: {e}")
                return None
        return response.choices[0].message.content



    def ask_ollama(self, question, image=None):
        image_bytes = []
        for img in image:
            image_bytes.append(image2base64_ollama(img))

        response = chat(
            model='llava',
            messages=[
                {
                    'role': 'user',
                    'content': question,
                    'images': image_bytes,
                }
            ],
        )

        return response.message.content
