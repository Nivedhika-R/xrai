from threading import Event
from queue import Queue
from utils import Frame

class QueryCollector:
    def __init__(self, max_queue_size=5):
        self.latest_frames_queue = Queue(max_queue_size)
        self.pending_questions_queue = Queue(max_queue_size)

        self.new_question_event = Event()

    def wait_for_question(self):
        self.new_question_event.wait()

    def clear_new_question_event(self):
        self.new_question_event.clear()

    # def add_question(self, question: Question):
    #     if self.pending_questions_queue.full():
    #         self.pending_questions_queue.get()

    #     self.pending_questions_queue.put(question)
    #     self.new_question_event.set()

    def add_frame(self, frame: Frame):
        if self.latest_frames_queue.full():
            self.latest_frames_queue.get()

        self.latest_frames_queue.put(frame)

    # def get_latest_question(self) -> Question:
    #     if not self.pending_questions_queue.empty():
    #         return self.pending_questions_queue.queue[-1]
    #     return None

    def get_latest_n_frames(self, n):
        frames = []
        for i in range(min(n, self.latest_frames_queue.qsize())):
            frames.append(self.latest_frames_queue.queue[-i-1])
        return frames

    def get_latest_query(self):
        latest_frame = None
        if not self.latest_frames_queue.empty():
            latest_frame = self.latest_frames_queue.queue[-1]
        return self.get_latest_question(), latest_frame
