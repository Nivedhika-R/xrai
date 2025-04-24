import time
import numpy as np
import cv2

from chatgpt_helper import ChatGPTHelper

class TutorialFollower:
    def __init__(self, frame_deque, yolo, instructions_path="instrs_and_inputs", task="snap-circuit"):
        self.instructions_path = instructions_path
        self.task = task
        self.yolo = yolo

        self.frame_deque = frame_deque
        self.chat_gpt = ChatGPTHelper()

        self.answer =  None
        self.current_instruction = ""
        self.current_instruction_index = 0
        self.task = "snap-circuit" #"humidifier"
        self.all_objects = {} # instruction: [object1, object2, ...]

    def get_current_objects(self):
        return self.all_objects[self.current_instruction]

    #Break instructions down into bite size steps
    def instruction_breakdown(self, instructions):
        prompt = "Can you break these intructions down into detailed steps of how to do each of the instruction? Assume all the components needed for the instructions are available. You do not need to give me too obvious or overly simplified steps. Give me each step as a bullet point list. Do not add extra lines or titles or details. If an instruction gives an option of two methods, just pick one and make sure the rest of the ingredients are consistant with it. Do not give me extra information or optional steps. Instructions: "+ instructions
        return self.chat_gpt.ask_gpt_3_5(prompt).splitlines()

    def get_curr_instruction(self, frames, instructions):
        prompt = "Provided is a list of instructions to perform a task. Look at the ego-centric images that show the last 10 seconds of what I have been doing from my headmounted device and tell me which step I should do next, that is, what is the instruction I should currently follow. Give me the instruction as an instruction number and nothing else in the format: 'Instruction number: <instruction>', with the first instruction being instruction 1. If you don't have an answer, answer with instruction 1. Make this inference based on what you see as the state of my environment in the image. Also tell me what objects in my image I need to do this instruction. Tell me in the format: 'Needed objects: <list of objects>'. Here are the instructions:" + str(instructions)
        return self.chat_gpt.ask(prompt, frames)

    def is_instruction_complete(self, frames, instructions, current_instruction):
        prompt = "I am currently trying to do the instruction: " + current_instruction + "\n Have I done the instruction? I am giving you a frame showing the current state of my environment from an ego-centric view and the previous state. Does it look like the instruction may have been done? Be true with your answers, each piece needs to be in the location the instruction says. If you see the full snap circuit transparent board and you think it is likely that the step is done, be linient and say it is done. But if the full board is not visible in the image, do not assume the step is done. The board has each row named A-G top to bottom and 1-10 as columns left to right. Answer just True or False. If false, tell me what I am missing. Here is the complete list of instructions: " + str(instructions)
        return self.chat_gpt.ask(prompt, frames)

    def load_instructions(self, instruction_file, objects_file):
        file = open(instruction_file, "r")
        text = file.read()
        self.instructions = text.splitlines()

        file = open(objects_file, "r")
        text = file.read()
        objects = text.splitlines()

        for i in range(len(objects)):
            objects[i] = objects[i].split(",")
            for j in range(len(objects[i])):
                objects[i][j] = objects[i][j].strip()
            self.all_objects[self.instructions[i]] = objects[i]

    def start(self):
        self.load_instructions(f"{self.instructions_path}/{self.task}/instructions.txt", f"{self.instructions_path}/{self.task}/objects.txt")

        print("Instructions:")
        for instruction in self.instructions:
            print("-", instruction)
            print("  - Objects:", self.all_objects[instruction])

        self.current_instruction = self.instructions[0]
        self.start_following()

    def start_following(self):
        while True:
            while len(self.frame_deque) < 2:
                time.sleep(0.1)
            self.latest_frames = []
            self.latest_frames.append(self.frame_deque[-2])
            self.latest_frames.append(self.frame_deque[-1])
            frame_imgs = []
            if len(self.latest_frames) > 0:
                for frame in self.latest_frames:
                    frame_imgs.append(np.asarray(frame.img))
                try:
                    answer = self.is_instruction_complete(frame_imgs, self.instructions, self.current_instruction)
                except:
                    continue

                self.prev_instruction_index =  self.current_instruction_index
                for line in answer.splitlines():
                    if "true" in line.lower():
                        self.current_instruction_index += 1
                    if 'Instruction number:' in line:
                        current_instruction_index = line.split('Instruction number:')[1]
                        self.current_instruction_index = int(current_instruction_index)

                if  self.current_instruction_index < self.prev_instruction_index:
                    self.current_instruction_index = self.prev_instruction_index

                self.current_instruction = self.instructions[self.current_instruction_index]
                self.answer = self.current_instruction + '\n Current instruction state: ' + answer
                # if (self.prev_instruction_index != self.current_instruction_index) or self.current_instruction_index == 0:
                #     ferret_prompt = "Find the following object for me. Give me an answer as a comma separated list with the format: object name = <co-ordinates>, object name:<co-ordinates>. If you cannot find a certain object with confidence, replace its coordinates with None. Do not give me more coordinates than I have asked for. Only give me coordinated for the objects I asked. The objects are: " + str(self.inst_inputs[self.current_instruction_index])
                #     response = self.ferret_gpt.ask_llms(ferret_prompt, "", frame_imgs[-1], None, None)
                #     self.answer = str(last_frameID)+ "///" + self.current_instruction + '\n Current instruction state: ' + answer + response

            time.sleep(0.1)

    def get_answer(self):
        return self.answer

    def clear_answer(self):
        self.answer = None
        
        
    def run_object_detection(self, frame):
        object_labels = []
        object_centers = []
        object_confidences = []
        yolo_results = self.yolo.predict(frame.img)
        for result in yolo_results:
            object_labels.append(result["class_name"])
            bbox = result["bbox"]
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            object_centers.append((center_x, frame.img.shape[0] - center_y))
            object_confidences.append(result["confidence"])

        # # save image to disk (to debug)
        # os.makedirs("images", exist_ok=True)
        # img_path = os.path.join("images", f"image_c{frame.client_id}_{frame.timestamp}.png")
        # # draw bounding boxes
        for result in yolo_results:
            bbox = result["bbox"]
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame.img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame.img, result["class_name"], (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame.img
        # # save the image
        # logger.warning("Saving image to %s", img_path)
        # cv2.imwrite(img_path, frame.img)
