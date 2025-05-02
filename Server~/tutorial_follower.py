import re
import time
import json
import numpy as np
import cv2
from frame import Frame
from logger import logger

from chatgpt_helper import ChatGPTHelper
from constants import display_labels

class TutorialFollower:
    def __init__(self, frame_deque, yolo, board_tracker=None, instructions_path="instructions", task="snap-circuit"):
        self.instructions_path = instructions_path
        self.task = task
        self.yolo = yolo

        self.frame_deque = frame_deque
        self.chat_gpt = ChatGPTHelper()

        self.answer =  None
        self.current_instruction_index = 0
        self.task = "snap-circuit" #"humidifier"
        self.instructions = []
        self.additional_texts = []
        self.all_objects = {} # instruction: [object1, object2, ...]
        self.board_tracker = board_tracker

    def get_current_objects(self):
        return self.all_objects[self.instructions[self.current_instruction_index]]

    def ask_llm(self, images):
        #            "The third image is a frame from a previous viewpoint. " + \
        prompt = \
            "I am currently trying to do the current instruction: " + \
            self.instructions[self.current_instruction_index] + "\n" + \
            "Here is a some additional information: " + \
            self.additional_texts[self.current_instruction_index] + "\n" + \
            "Here is the complete list of instructions: " \
            + str(self.instructions) + "\n" + \
            "I am giving you four images (images may be a bit blurry and have some glare): " + \
            "The first image (with a black background), shows the current state of my environment from an ego-centric view. " + \
            "The second image is the same as the first image but with important objects labels. " + \
            "The third image is a sample image of the expected result of the instruction. " + \
            "ANSWER THIS QUESTION: Does it look like the current instruction has been done in the first image I sent? " + \
            "Be true with your answers, as each piece needs to be in the location the instruction says. " + \
            "If you see the full snap circuit transparent board and you think it is likely that the step is done, make sure the pieces are not placed in random positions, you mess this up too often so be careful about that. " + \
            "Sometimes images are over exposed and hard to understand the pieces, so look for colors and also similarity to the expected result. " + \
            "The board has rows labeled A to G from top to bottom and columns labeled 1 to 10 from left to right, written in black marker. " + \
            "The batteries may cover some of the bottom row labels. Go off of your spatial reasoning if the labels cant be seen. " + \
            "Your answer should have True or False as the first word. Look at the example image and try to match that, compare that to the first 3 frames I give you which is what I do. I highlighted the change you should be looking for" + \
            "If false, just tell me what I should do to complete the step and what I am missing or what I have done wrong. If true, just say 'True'. Keep responses brief (one sentence). " + \
            "Don't automatically skip steps if you haven't see the step happen, especially pay attention to placement. The location of parts is very important. Always give me an answer even if you are not sure. "

        # prompt = \
        #     "I am currently trying to do the current instruction: " + \
        #     self.instructions[self.current_instruction_index] + "\n" + \
        #     "Here is a some additional information: " + \
        #     self.additional_texts[self.current_instruction_index] + "\n" + \
        #     "I am giving you four images (images may be a bit blurry and have some glare): " + \
        #     "The first image (with a black background), shows the current state of my environment from an ego-centric view. " + \
        #     "The second image is the same as the first image but with important objects labels. " + \
        #     "The third image is a sample image of the expected result of the instruction. " + \
        #     "ANSWER THIS QUESTION: Does it look like the current instruction has been done in the first image I sent? " + \
        #     "Be true with your answers, as each piece needs to be in the location the instruction says. " + \
        #     "If you see the full snap circuit transparent board and you think it is likely that the step is done, make sure the pieces are not placed in random positions, you mess this up too often so be careful about that. " + \
        #     "Sometimes images are over exposed and hard to understand the pieces, so look for colors and also similarity to the expected result. " + \
        #     "The board has rows labeled A to G from top to bottom and columns labeled 1 to 10 from left to right, written in black marker. " + \
        #     "The batteries may cover some of the bottom row labels. Go off of your spatial reasoning if the labels cant be seen. " + \
        #     "Your answer should have True or False as the first word. Look at the example image and try to match that, compare that to the first 3 frames I give you which is what I do. I highlighted the change you should be looking for" + \
        #     "If false, just tell me what I should do to complete the step and what I am missing or what I have done wrong. If true, just say 'True'. Keep responses brief (one sentence). " + \
        #     "Don't automatically skip steps if you haven't see the step happen, especially pay attention to placement. The location of parts is very important. And don't say 'I don't know' or that you can't do it. Always give an answer. "

        ans = self.chat_gpt.ask(prompt, images)
        logger.info("\nAnswer from ChatGPT: " + ans + "/n")
        logger.info("\nCurrent Step: " + str(self.current_instruction_index) + "/n")
        return ans

    def load_instructions(self, instruction_file):
        instructions = json.load(open(instruction_file, "r"))
        for instruction in instructions:
            self.instructions.append(instruction["instruction"])
            self.additional_texts.append(instruction["additional_text"])
            self.all_objects[instruction["instruction"]] = instruction["objects"]

    def start(self):
        self.load_instructions(f"{self.instructions_path}/{self.task}/instructions.json")

        logger.info("Instructions:")
        for instruction in self.instructions:
            logger.info(f"- {instruction}")
            logger.info(f"  - Objects: {self.all_objects[instruction]}")

        self.start_following()

    def start_following(self):
        while True:
            while len(self.frame_deque) < 2:
                time.sleep(0.1)

            latest_frame = self.frame_deque[-1]
            previous_frame = self.frame_deque[-2]
            # latest_frame_with_bboxes = self.run_object_detection(latest_frame)
            # get sample image
            sample_image_path = f"{self.instructions_path}/{self.task}/images/step{self.current_instruction_index}.jpg"
            sample_image = cv2.imread(sample_image_path)
            sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
            sample_image = cv2.resize(sample_image, (1280, 720))
            sample_frame = Frame(1, sample_image, None, None, None)

            images = []
            cropped_board = self.board_tracker.get_board_segment(latest_frame.img.copy())
            images.append(cropped_board)
            # images.append(latest_frame.img.copy())
            cropped_frame = Frame(latest_frame.client_id, cropped_board, latest_frame.cam_mat, latest_frame.proj_mat, latest_frame.dist_mat, time.time())
            cropped_board_with_bboxes = self.run_object_detection(cropped_frame)
            images.append(cropped_board_with_bboxes.img.copy())
            # images.append(previous_frame.img.copy())
            images.append(sample_frame.img.copy())
            try:
                answer = self.ask_llm(images)
            except Exception as e:
                logger.error(f"Error while asking LLM: {e}", exc_info=True)
                continue

            self.prev_instruction_index =  self.current_instruction_index
            for line in answer.splitlines():
                if "true" in line.lower():
                    self.current_instruction_index += 1
                if 'Instruction number:' in line:
                    current_instruction_index = line.split('Instruction number:')[1]
                    self.current_instruction_index = int(current_instruction_index)

            if self.current_instruction_index < self.prev_instruction_index:
                self.current_instruction_index = self.prev_instruction_index

            match = re.match(r'^(True|False)[\s,\.]*(.*)', answer.strip())
            if match:
                first_part, rest = match.groups()
                current_instruction_state = (first_part == "True")
                # capitalize first letter
                final_part = rest.strip()
                if final_part:
                    final_part = final_part[0].upper() + final_part[1:]
            else:
                # fallback
                current_instruction_state = False
                final_part = answer.strip().capitalize()

            current_instruction = self.instructions[self.current_instruction_index]
            self.answer = ("Step completed!" if current_instruction_state else "Step not completed yet.") + "\n\n" + \
                            "Current Step: " + current_instruction + "\n\n" + \
                            "Feedback: " + "\"" + final_part + "\""

            time.sleep(0.1)

    def get_answer(self):
        return self.answer

    def clear_answer(self):
        self.answer = None

    def run_object_detection(self, frame):
        yolo_results = self.yolo.predict(frame.img)

        # # save image to disk (to debug)
        # os.makedirs("images", exist_ok=True)
        # img_path = os.path.join("images", f"image_c{frame.client_id}_{frame.timestamp}.png")

        # draw bounding boxes
        frame_bbox = Frame(frame.client_id, frame.img.copy(), frame.cam_mat, frame.proj_mat, frame.dist_mat, time.time())
        for result in yolo_results:
            if result["class_name"] not in self.get_current_objects():
                continue
            bbox = result["bbox"]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_bbox.img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame_bbox.img, display_labels[result["class_name"]], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # save the image for debugging
        cv2.imwrite("yolo.png", frame_bbox.img)
        return frame_bbox
