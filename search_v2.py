# Author: Muhammed Elyamani
# Date: 25/01/2023
# GitHub: https://github.com/WikiGenius

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.core.window import Window
from kivy.lang import Builder
import cv2
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import asone
from asone import ASOne
import imutils
import time
from utils import draw_boxes

Builder.load_file('searchv2.kv')
Window.size = (350, 600)

class SearchDashboard(BoxLayout):
    pass


class SearchApp(App):
    def build(self):
        Window.clearcolor=(0,0,0.2)
        self.search = SearchDashboard()
        vid_path = 'video1.mp4'
        self.capture = cv2.VideoCapture(vid_path)
        # load model
        self.detector = ASOne(detector=asone.YOLOV8N_PYTORCH, weights=None, use_cuda=True)
        
        # text input variable
        self.filter_classes = None
        
        Clock.schedule_interval(self.update, 1/33)
        return self.search
    
    def update_search(self):
        self.filter_classes = self.search.text_input.text
        if self.filter_classes:
            self.filter_classes = self.filter_classes.split(',')

    def detect_object(self, img):
        prevTime = time.time()
        dets, img_info =  self.detector.detect(img, conf_thres=0.25, iou_thres=0.45)
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        
        if dets is not None: 
            bbox_xyxy = dets[:, :4]
            # scores = dets[:, 4]
            class_ids = dets[:, 5]
            img = draw_boxes(img, bbox_xyxy, class_ids=class_ids, filter_classes=self.filter_classes)

        img = draw_boxes(img, bbox_xyxy, class_ids, filter_classes=self.filter_classes)
        
        cv2.line(img, (20, 25), (127, 25), [85, 45, 255], 30)
        cv2.putText(img, f'FPS: {int(fps)}', (11, 35), 0, 1, [
                    225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        
        return img
        
        
    def update(self, *args):
        ret, frame = self.capture.read()
        if not ret:
            return
        frame = imutils.resize(frame, width=600)
        frame = self.detect_object(frame)
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size = (frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.search.image.texture = img_texture

if __name__ == '__main__':
    SearchApp().run()
