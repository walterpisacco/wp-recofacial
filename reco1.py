import cv2
import imutils
import face_recognition
import pickle
#from vidgear.gears import CamGear
from gtts import gTTS
from flask import Flask, render_template, json, request
from datetime import datetime
import wget
#pip3 install flask-mysql
from flaskext.mysql import MySQL

app = Flask(__name__ , template_folder= "modulos")
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

mysql = MySQL()

from config import config
configuracion = config()

# MySQL configurations
app.config['MYSQL_DATABASE_USER'] = configuracion.usuario
app.config['MYSQL_DATABASE_PASSWORD'] = configuracion.password
app.config['MYSQL_DATABASE_DB'] = configuracion.base
app.config['MYSQL_DATABASE_HOST'] = configuracion.servidor

mysql.init_app(app)

#busco el cliente de una sesion de usuario de la plataforma conectada desde esta ip
conn = mysql.connect()
cursor = conn.cursor()

query = "select clave,valor from sas_admin.configs where id_client = 1"
cursor.execute(query)
conn.commit()
results = cursor.fetchall()
if len(results) > 0:
    for row in results:
        if row[0] == 'desde':
            desde = row[1]

        if row[0] == 'hasta':
            hasta = row[1]            
        
        desde = datetime.strptime(desde, "%Y-%m-%d %H:%M")
        hasta = datetime.strptime(hasta, "%Y-%m-%d %H:%M")

url = configuracion.pathDescarga
wget.download(url, 'models/faces.pickle')

video = cv2.VideoCapture(int(configuracion.camara2))

app.ban = False #Bandera Para que cargue una sola Vez el Modelo
app.detector = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
app.data = pickle.loads(open("models/faces.pickle", "rb").read())


while True:
    success, image = video.read()
    if success:
        now = datetime.now()
        if desde <= now <= hasta:
            print ("procesa")
                #image = imutils.resize(image, width=720)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grey scale

            faces = app.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            color = (255, 255, 255)
            for (x,y,x1,y1) in faces:
                cv2.rectangle(image,(x,y),(x1+x,y1+y),color,2)

            boxes = [(y, x + x1, y + y1, x) for (x, y, x1, y1) in faces]
            
            encodings = face_recognition.face_encodings(image, boxes)
            names = []        

            for encoding in encodings: 
                matches = face_recognition.compare_faces(app.data["encodings"],encoding,tolerance=0.5)
                name = "Desconocido"      
                color = (0, 255, 0)
                if True in matches:
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    for i in matchedIdxs:
                        name = app.data["names"][i]
                        counts[name] = counts.get(name, 0) + 1
                    name = max(counts, key=counts.get)
                else:
                    color = (0, 255, 0)
                names.append(name)
                if name!="Desconocido":
                    if app.ban == False:
                        print(name)
                        app.ban = True
                else:
                    app.ban = False
        else:
            print ("NO procesa")
     
    cv2.imshow('image', image)
    if cv2.waitKey(1) == ord('q'):
        break           

video.release()
cv2.destroyAllWindows()

