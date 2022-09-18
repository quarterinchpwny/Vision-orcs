from flask import Flask, render_template, request, make_response
import os
import cv2
import pytesseract
import service
import pandas as pd
import pkg_resources
pytesseract.pytesseract.tesseract_cmd=r'Tesseract/tesseract.exe'
#pytesseract.pytesseract.tesseract_cmd=r'C:/Program Files/Tesseract-OCR/tesseract.exe'
app = Flask(__name__)
@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/galleryPlain", methods=['GET', 'POST'])
def gallery():
    return render_template('gallery_Plain.html')

@app.route("/galleryRuled", methods=['GET', 'POST'])
def galleryRuled():
    return render_template('gallery_Ruled.html')

@app.route("/startpage", methods=['GET', 'POST'])
def startpage():
    return render_template('startpage.html')

@app.route("/ruleplain", methods=['GET', 'POST'])
def sruleplain():
    return render_template('ruleplain.html')



@app.route("/cam", methods=['GET', 'POST'])
def cam():
    return render_template('cam.html')

@app.route("/predictRuled", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
        image = cv2.imread(file_path)
        print(image)


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray',gray)
        cv2.waitKey()
        
        saltpep = cv2.fastNlMeansDenoising(gray,None,9,13)
        # original_resized = cv2.resize(saltpep, (0,0), fx=.2, fy=.2)
        cv2.imshow('Grayscale',saltpep)
        cv2.waitKey()

        #blur
        blurred = cv2.GaussianBlur(saltpep, (3, 3), 0)  
        # original_resized = cv2.resize(blured, (0,0), fx=.2, fy=.2)
        cv2.imshow('blured',blurred)
        cv2.waitKey()
        thresh1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        # original_resized = cv2.resize(thresh, (0,0), fx=.2, fy=.2)
        cv2.imshow('Threshold',thresh1)
        cv2.waitKey()

        dilation = cv2.erode(thresh1, rect_kernel, iterations = 1)
        cv2.imshow('dilation',dilation)
        cv2.waitKey()
        # original_resized = cv2.resize(img_dilation, (0,0), fx=.2, fy=.2)
       
       
        

        text = pytesseract.image_to_string(thresh1, lang='eng+train',config='--psm 3 --oem 3 -c preserve_interword_spaces=1 ')
        res = pytesseract.image_to_data(dilation, lang='eng+train', output_type='data.frame',config='--psm 6 --oem 3 -c preserve_interword_spaces=1 ')

        res = res[res.conf != -1]
       
        print(res[['conf','text']])
       
        lines = res.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'] \
                                     .apply(lambda x: ' '.join(list(x))).tolist()
        confs = res.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()
    
        line_conf = []
    
        for i in range(len(lines)):
            if lines[i].strip():
                line_conf.append((lines[i], round(confs[i],3)))
        print(line_conf)

        splits = text.splitlines()
        s = "\n".join(splits)
        #s = "<br>".join(splits)
        s = s.replace("£","{")
        s = s.replace("“","\"")
        s = s.replace("’","\"")
        s = s.replace("‘","\"")
        s = s.replace("€","{")
        s = s.replace("§","+")
        s = s.replace("»","=")
        s = s.replace("«","*")
        
        s = s.replace("print In","println")
        s = s.replace("printin","println")
        s = s.replace("printin","println")
        s = s.replace("printin","println")
        s= s+"}"
        print(splits)
        print(s)
        return render_template('sec.html', pred_output=s, user_image=file_path)

@app.route("/predictPlain", methods=['GET', 'POST'])
def unruled():
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
        image = cv2.imread(file_path)
        print(image)


        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('gray',gray)
        cv2.waitKey() 

        # Draw bounding boxes
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        cv2.imshow('thresh',thresh)
        cv2.waitKey() 
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

        cv2.imshow('image',image)
        cv2.waitKey() 
        # OCR
        text = pytesseract.image_to_string(255 - thresh, lang='eng',config='--psm 3 --oem 3 -c preserve_interword_spaces=1 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890=+-_(){}')
        print(text)


        res = pytesseract.image_to_data(image, output_type='data.frame',config='--psm 3 --oem 3 -c preserve_interword_spaces=1 tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890=+-_(){}')
        res = res[res.conf != -1]
        print(res[['conf','text']])
        lines = res.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'] \
                                     .apply(lambda x: ' '.join(list(x))).tolist()
        confs = res.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()
    
        line_conf = []
    
        for i in range(len(lines)):
            if lines[i].strip():
                line_conf.append((lines[i], round(confs[i],3)))
        print(line_conf)

        splits = text.splitlines()
        s = "\n".join(splits)
        #s = "<br>".join(splits)
        s = s.replace("£","{")
        s = s.replace("“","\"")
        s = s.replace("’","\"")
        s = s.replace("‘","\"")
        s = s.replace("€","{")
        s = s.replace("§","+")
        s = s.replace("»","=")
        s = s.replace("«","*")
        s = s.replace("print In","println")
        s = s.replace("printin","println")

       
        print(splits)
        print(s)
  
    
   
 
        return render_template('sec.html', pred_output=s, user_image=file_path)

@app.route('/capture_img', methods=['POST'])
def capture_img():
    msg = service.save_img(request.form["img"])
    return make_response(msg)

if __name__ == "__main__":
    app.run(threaded=False,debug=True)