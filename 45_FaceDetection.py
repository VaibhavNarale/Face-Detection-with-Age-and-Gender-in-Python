import cv2

video = cv2.VideoCapture(0)

def faceBox(faceNet,frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227,227), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    box=[]
    for i in range(detection.shape[2]):
        confidence = detection[0,0,i,2]
        if confidence>0.7:
            x1 = int(detection[0,0,i,3]*frameWidth)
            y1 = int(detection[0,0,i,4]*frameHeight)
            x2 = int(detection[0,0,i,5]*frameWidth)
            y2 = int(detection[0,0,i,6]*frameHeight)
            box.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,295,0),1)
    return frame,box
# define Model

face_pbtxt = r'C:\Users\Hp\Desktop\Udemy\Models\opencv_face_detector.pbtxt'
face_txt= r'C:\Users\Hp\Desktop\Udemy\Models\opencv_face_detector_uint8.pb'
age_protxt = r'C:\Users\Hp\Desktop\Udemy\Models\age_deploy.prototxt'
age_model = r'C:\Users\Hp\Desktop\Udemy\Models\age_net.caffemodel'
gender_protext = r'C:\Users\Hp\Desktop\Udemy\Models\gender_deploy.prototxt'
gender_model = r'C:\Users\Hp\Desktop\Udemy\Models\gender_net.caffemodel'

# LOad the model 
faceNet = cv2.dnn.readNet(face_txt,face_pbtxt)
ageNet = cv2.dnn.readNet(age_model,age_protxt)
genderNet = cv2.dnn.readNet(gender_model,gender_protext)

# age classificatrion
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_classification = ['(0-2)','(4-6)','(8-12)','(15-20)','(21-24)','(25-32)','(38-43)','(48-53)','(54-58)','(60-100)'] 
gender_classification = ['(Male)','(Female)']

while True:
    ret,frame= video.read()
    frame,box = faceBox(faceNet, frame)
    for b in box:
        face = frame[b[1]:b[3], b[0]:b[2]]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderpro = genderNet.forward()
        genders = gender_classification[genderpro[0].argmax()]
        
        ageNet.setInput(blob)
        agepro=ageNet.forward()
        ages=age_classification[agepro[0].argmax()]        
        
        label="{},{}".format(genders,ages)
        cv2.rectangle(frame, (b[0], b[1]-30), (b[2], b[1]), (0, 255, 0), -1)
        cv2.putText(frame, label, (b[0], b[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Age_gender", frame)  # Correct usage
    k=cv2.waitKey(9)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()


'''
If the model detects 3 faces, the output might look like this:

python
Copy
Edit
          fix  fix  confid x1  y1   x2   y2
array([[[[0.0, 0.0, 0.95, 0.2, 0.3, 0.5, 0.6],   # Face 1
         [0.0, 0.0, 0.85, 0.4, 0.2, 0.7, 0.5],   # Face 2
         [0.0, 0.0, 0.78, 0.6, 0.1, 0.8, 0.4]]]]) # Face 3
Explanation:
Face 1:

Confidence: 0.95 (95% chance it's a face)

Bounding box:

Top-left: (0.2, 0.3) (normalized)

Bottom-right: (0.5, 0.6) (normalized)

Face 2:

Confidence: 0.85 (85% chance)

Bounding box:

Top-left: (0.4, 0.2)

Bottom-right: (0.7, 0.5)

Face 3:

Confidence: 0.78 (78% chance)

Bounding box:

Top-left: (0.6, 0.1)

Bottom-right: (0.8, 0.4)
'''

'''
 Structure of the Last Dimension (7 values):
Index	Description
0	Not used (always 0)
1	Not used (always 0)
2	Confidence Score
3	x1 (Top-left x-coordinate)
4	y1 (Top-left y-coordinate)
5	x2 (Bottom-right x-coordinate)
6	y2 (Bottom-right y-coordinate)
'''