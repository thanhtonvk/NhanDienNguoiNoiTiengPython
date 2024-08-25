import os
from flask import Flask, render_template, request, redirect
import cv2
import unidecode
from dal.NguoiDungDalSqlite import NguoiDungDal
from modules.face_detection import FaceDetection
from modules.face_recognition import FaceRecognition
app = Flask(__name__)

nguoiDungDal = NguoiDungDal()
faceDetector = FaceDetection()
faceRecognition = FaceRecognition()

import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.route('/nguoi-dung/them', methods=['GET', 'POST'])
def them_sv():
    if request.method == 'GET':
        return render_template('them_nguoi_dung.html')
    ho_ten = request.form['ho_ten']
    id = request.form['id']
    f = request.files['file']
    save_path = f'static/faces/image.png'
    try:
        f.save(save_path)
    except:
        os.remove(save_path)
        f.save(save_path)
    image = cv2.imread(save_path)
    result = faceDetector.save_face(id,image)
    os.remove(save_path)
    if result>0:
        image_face = cv2.imread(f"static/faces/{id}/face.png")
        emb = faceRecognition.get_embed(image_face)
        nguoiDungDal.insert(ho_ten, id,emb)
        return redirect('/danh-sach') 
    return render_template('them_nguoi_dung.html')


@app.route('/nguoi-dung/xoa/<int:id>', methods=['GET'])
def xoa(id):
    nguoiDungDal.delete(id)
    return redirect('/danh-sach')


@app.route('/danh-sach', methods=['GET'])
def danh_sach_sv():
    NguoiDungs =nguoiDungDal.get()
    print(NguoiDungs)
    return render_template('danh_sach_nguoi_dung.html', NguoiDungs=NguoiDungs)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html', data = None)
    
    nguoi_dungs = nguoiDungDal.get()
    f = request.files['image']
    save_path = f'static/image.png'
    f.save(save_path)
    image = cv2.imread(save_path)
    predict = faceDetector.detect(image)
    boxes = predict['boxes']
    faces = predict['faces']
    list_kq = []
    for idx, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)
                face = faces[idx]
                nguoi_dung = faceRecognition.search_face(face, nguoi_dungs)
                if nguoi_dung is not None:
                    list_kq.append(nguoi_dung)
                    
                    cv2.putText(image, f"{unidecode.unidecode(nguoi_dung.HoTen)}",
                                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, f"Khong the nhan dien",
                                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(save_path,image)
    image_base64 = encode_image(save_path)
    if len(list_kq)>0:
        strResult =""
        for i in list_kq:
            strResult+=i.HoTen+"; "
    else:
        strResult = "Không thể nhận diện"
    response = {'image_path': image_base64,
                'result': strResult, 'type': 1}
    return render_template('index.html', data=response)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
