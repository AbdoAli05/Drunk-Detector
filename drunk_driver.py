from tensorflow.keras.preprocessing.image import ImageDataGenerator
from kivy.uix.boxlayout import BoxLayout
from kivy.lang import Builder
from kivy.clock import Clock
from tensorflow import keras
from kivy.app import App
import tensorflow as tf
import glob
import time
import cv2
import os
import threading
from kivy.uix.gridlayout import GridLayout

from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import ScreenManager, Screen
from functools import partial
from kivy.uix.boxlayout import BoxLayout


class MainScreen(Screen):
    def changeStat(self):
        self.ids.qes.opacity = 0
        self.ids.qes.height = 0
        self.ids.ans.opacity = 0
        self.ids.ans.height = 0
        self.ids.pb.value = 0.50
        self.ids.pbv.text = "50% Sober"
        self.ids.sober.background_color = '#1cff00'
        self.ids.drunk.background_color = 0.231, 0.238, 0.244, 1


class Manager(BoxLayout):
    pass


Builder.load_string('''
<MainScreen>:
    BoxLayout:
        orientation: 'vertical'
        Image:
            id: vid
            resolution: (640, 480)
    
        Button:
            id: drunk
            text: 'Drunk'
            size_hint_y: None
            height: '48dp'
            background_color:0.231,0.238,0.244,1
        Button:
            id: sober
            text: 'Sober'
            size_hint_y: None
            height: '48dp'
            background_color:0.231,0.238,0.244,1
        
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: None
            height: '25dp'
            padding:[dp(20),dp(10),dp(20),dp(10)] # padding: [dp(20),dp(10),dp(20),dp(10)]         
            ProgressBar:
                id: pb
                min:0
                max:1
                pos_hint: {'x':.1}
                size_hint:(.70,1.0) 
                value:0
            Label:
                id: pbv
                text: '  %'
                pos_hint: {'x':.90}
                size_hint:(.15,1.0)  
                
            
        Button:
            orientation: 'horizontal'
            id: qes
            text: 'Touch Yellow Button !?'
            size_hint_y: None
            height: '40'
            opacity: 0
            background_color:0.231,0.238,0.244,1
    
        BoxLayout:
            id:ans
            orientation: 'horizontal'
            size_hint_y: None
            height: '0'        
            opacity: 0
            Button:
                id: ansA
                on_press : root.changeStat()
                text: 'Yellow'
                size_hint:(.33,1.0) 
                height: '48dp'
                background_color:'#Fff400'
            Button:
                id: ansB
                text: 'Blue'
                size_hint:(.33,1.0) 
                height: '48dp'
                background_color:'#0018ff'
   
            Button:
                id: ansC
                text: 'White'
                size_hint:(.33,1.0) 
                height: '48dp'
                background_color:'#Ffffff'
    

''')

#
# class CameraClick(BoxLayout):
#
#     # def run(self):
#     #     for _ in range(10000000000):
#     #         self.capture()
#     #         # self.ids.drunk.on_press=True
#     #         time.sleep(.10)
#
#     def predict(self):
#         drunk = 0
#         sober = 0
#         f = glob.glob('test/*')
#         if len(f) < 1:
#             return False, 0, 0
#         test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
#             .flow_from_directory(directory='test/', target_size=(224, 224), classes=[''], shuffle=False)
#         predicted = loaded_model.predict(test_batches)
#         for a in predicted:
#             drunk += a[0]
#             sober += a[1]
#         print("drunk", drunk / len(predicted))
#         print("sober", sober / len(predicted))
#
#         stat = drunk > sober
#         return stat, drunk, sober
#
#     def extract_faces(self):
#         self.imgList = []
#         image = glob.glob('*.png')
#         for i in image:
#             imgss = cv2.imread(i)
#             self.imgList.append(imgss)
#
#         for img in self.imgList:
#             face = []
#             faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#             faces = faceCascade.detectMultiScale(
#                 img,
#                 scaleFactor=1.3,
#                 minNeighbors=2,
#                 minSize=(70, 70)
#             )
#             for (x, y, w, h) in faces:
#                 face = img[y:y + h, x:x + w]
#             # faces_list.append(face)
#             if len(face) > 0:
#                 cv2.imwrite('test/' + str(self.uu) + 'a.jpg', face)
#                 self.uu += 1
#
#     def capture(self):
#         self.imgList = []
#         while True:
#             camera = self.ids['camera']
#             if self.count % 2 == 0:
#                 camera.export_to_png("IMG_{}.png".format(self.count))
#
#             if self.count > 6:
#                 self.extract_faces()
#                 self.stat, d, s = self.predict()
#                 if d == 0:
#                     break
#                 self.drunk += d
#                 self.sober += s
#                 self.uu = 0
#
#                 self.drunk = 0
#                 self.sober = 0
#                 self.count = -1
#                 if self.drunk > self.sober:
#                     self.stat = True
#                 self.changeStat()
#
#             self.count += 1
#             if cv2.waitKey(1) == ord('q'):
#                 break
#             break
#         print("Captured")


class Drunk_Driver(App):
    drunk, sober, count, uu = 0, 0, 0, 0
    imgList = []
    stat = False
    count = 0

    def build(self):
        threading.Thread(target=self.doit, daemon=True).start()
        sm = ScreenManager()
        self.main_screen = MainScreen()
        sm.add_widget(self.main_screen)
        return sm



    def doit(self):

        self.do_vid = True

        # cam = cv2.VideoCapture(0)
        cam = cv2.VideoCapture("VideoTrimmer_Adidas CEO Herbert Hainer_ How I Work.mp4")

        while self.do_vid:
            drunk=0
            sober=0
            len_pro=0
            self.count += 1
            ret, frame = cam.read()
            uu = 0
            if self.count % 30 == 0:
                cv2.imwrite(str(self.count) + 'img.jpg', frame)
            if self.count > 600:
                self.count = 0
                image = glob.glob('*.jpg')
                imgList = []
                for i in image:
                    images = cv2.imread(i)
                    imgList.append(images)

                for img in imgList:
                    face = []
                    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                    faces = faceCascade.detectMultiScale(
                        img,
                        scaleFactor=1.3,
                        minNeighbors=2,
                        minSize=(70, 70)
                    )
                    for (x, y, w, h) in faces:
                        face = img[y:y + h, x:x + w]
                    # faces_list.append(face)

                    if len(face) > 0:
                        cv2.imwrite('test/' + str(uu) + 'a.jpg', face)
                        uu += 1

                f = glob.glob('test/*')
                if len(f) < 1:
                    return False, 0, 0
                test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
                    .flow_from_directory(directory='test/', target_size=(224, 224), classes=[''], shuffle=False)
                predicted = loaded_model.predict(test_batches)
                for a in predicted:
                    drunk += a[0]
                    sober += a[1]
                print("drunk", drunk / len(predicted))
                print("sober", sober / len(predicted))
                if drunk > sober:
                    self.stat = True
                    len_pro = drunk / len(predicted)
                else:
                    self.stat = False
                    len_pro= sober / len(predicted)

            list = [frame, self.stat, self.count, len_pro]

            Clock.schedule_once(partial(self.display_frame, list))

            cv2.waitKey(1)
        cam.release()
        cv2.destroyAllWindows()

    def stop_vid(self):
        # stop the video capture loop
        self.do_vid = False

    def display_frame(self, list, dt):
        # display the current video frame in the kivy Image widget
        # create a Texture the correct size and format for the frame
        # copy the frame data into the texture
        # flip the texture (otherwise the video is upside down)
        # actually put the texture in the kivy Image widget

        try:
            frame = list[0]
            stat = list[1]

            # print('stat = ', stat)
            if list[2] == 0:
                if not stat:
                    self.main_screen.ids.qes.opacity = 0
                    self.main_screen.ids.qes.height = 0
                    self.main_screen.ids.ans.opacity = 0
                    self.main_screen.ids.ans.height = 0
                    self.main_screen.ids.pb.value = list[3]
                    self.main_screen.ids.pbv.text = str(round(list[3]*100, 2))+"% Sober"
                    self.main_screen.ids.sober.background_color = '#1cff00'
                    self.main_screen.ids.drunk.background_color = 0.231, 0.238, 0.244, 1

                else:
                    self.main_screen.ids.ans.opacity = 1
                    self.main_screen.ids.ans.height = 48
                    self.main_screen.ids.qes.height = 48
                    self.main_screen.ids.qes.opacity = 1
                    self.main_screen.ids.pb.value = list[3]
                    self.main_screen.ids.pbv.text = str(round(list[3]*100, 2))+"% Drunk"
                    self.main_screen.ids.drunk.background_color = '#c40b0e'
                    self.main_screen.ids.sober.background_color = 0.231, 0.238, 0.244, 1

            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')

            # copy the frame data into the texture
            texture.blit_buffer(frame.tobytes(order=None), colorfmt='bgr', bufferfmt='ubyte')

            # flip the texture (otherwise the video is upside down
            texture.flip_vertical()
            # actually put the texture in the kivy Image widget
            self.main_screen.ids.vid.texture = texture

        except:
            print("Video Ended")
            cv2.destroyAllWindows()
            quit()


loaded_model = keras.models.load_model('Tue_MobileNetV2_2.h5')

if __name__ == '__main__':
    Drunk_Driver().run()
