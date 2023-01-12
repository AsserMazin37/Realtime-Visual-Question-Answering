from transformers import ViltProcessor, ViltForQuestionAnswering
import cv2
from threading import Thread

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.75
question_origin = (0, 20)
ans_origin = (0, 50)
thickness = 2
question = ''

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True

def answer_question(image, question):
    # prepare inputs
    encoding = processor(image, question, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)

    logits = outputs.logits
    idx = logits.argmax(-1).item()
    predicted_answer = model.config.id2label[idx]

    return predicted_answer

def input_question():
    global question
    question = input("Question: ")

def threadVideoGet(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """
    video_getter = VideoGet(source).start()

    while True:
        if question != 'end':
            input_thread = Thread(target=input_question, args=(), daemon=True)
            input_thread.start()

        # ESC is pressed
        if (cv2.waitKey(1) == 27) or video_getter.stopped or question == 'end':
            video_getter.stop()
            break
        frame = video_getter.frame
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if question.strip():
            cv2.putText(image, question, question_origin, font, 
                        font_scale, (255, 0, 0), thickness, cv2.LINE_AA)
                
            predicted_answer = answer_question(image, question)
                
            cv2.putText(image, predicted_answer, ans_origin, font, 
                        font_scale, (0, 0, 255), thickness, cv2.LINE_AA)

        cv2.imshow('VQA Demo', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.setWindowProperty('VQA Demo', cv2.WND_PROP_TOPMOST, 1) # keep window always open

    cv2.destroyAllWindows()

if __name__ == "__main__":
    threadVideoGet()