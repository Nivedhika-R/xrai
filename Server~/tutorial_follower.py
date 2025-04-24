import time
import numpy as np

from chatgpt_helper import ChatGPTHelper

class TutorialFollower:
    def __init__(self, frame_deque, instructions_path="instrs_and_inputs", task="snap-circuit"):
        self.instructions_path = instructions_path
        self.task = task

        self.frame_deque = frame_deque
        self.chat_gpt = ChatGPTHelper()

        self.answer =  None
        self.current_instruction = ""
        self.current_instruction_index = 0
        self.all_objects = {} # instruction: [object1, object2, ...]

    #Break instructions down into bite size steps
    def instruction_breakdown(self, instructions):
        prompt = "Can you break these intructions down into detailed steps of how to do each of the instruction? Assume all the components needed for the instructions are available. You do not need to give me too obvious or overly simplified steps. Give me each step as a bullet point list. Do not add extra lines or titles or details. If an instruction gives an option of two methods, just pick one and make sure the rest of the ingredients are consistant with it. Do not give me extra information or optional steps. Instructions: "+ instructions
        return self.chat_gpt.ask_gpt_3_5(prompt).splitlines()

    def get_curr_instruction(self, frames, instructions):
        prompt = "Provided is a list of instructions to perform a task. Look at the ego-centric images that show the last 10 seconds of what I have been doing from my headmounted device and tell me which step I should do next, that is, what is the instruction I should currently follow. Give me the instruction as an instruction number and nothing else in the format: 'Instruction number: <instruction>', with the first instruction being instruction 1. If you don't have an answer, answer with instruction 1. Make this inferrence based on what you see as the state of my environment in the image. Also tell me what objects in my image I need to do this instruction. Tell me in the format: 'Needed objects: <list of objects>'. Here are the instructions:" + str(instructions)
        return self.chat_gpt.ask(prompt, frames)

    def is_instruction_complete(self, frames, instructions, current_instruction):
        prompt = "I am currently trying to do the instruction: " + current_instruction + "\n Have I done the instruction? I am giving you a frame showing the current state of my environment from an ego-centric view. Does it look like the instruction may have been done? If there is any chance it might be done, say true. Be lenient in your responses. Answer just True or False. If false, tell me what I am missing. If unsure, say 'true'. Remember right is left and left is right (the image is mirrored). Here is the complete list of instructions: " + str(instructions)
        return self.chat_gpt.ask(prompt, frames[0])

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
