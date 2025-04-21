from logger import logger


class Debugger:
    def __init__(self, tutorial_follower):
        self.tutorial_follower = tutorial_follower
        return
    
    def start(self, ):
        while True:
            user_input = input()
            if user_input == "n":
                if self.tutorial_follower != None:
                    self.tutorial_follower.current_instruction_index += 1
                    logger.warning("Manual Override: Next instruction")
                    