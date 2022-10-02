import cv2
from deepface import DeepFace
import numpy as np
from csv import writer
import pandas as pd
import argparse
from pathlib import Path
from distutils.util import strtobool
from audio_analysis import audio_analyse

def main(args):

    # Write a new csv file for output
    column_names = ['label', 'happy', 'angry', 'disgust', 'sad', 'fear', 'neutral', 'surprise']

    file_path = Path("./train_data.csv")
    if not file_path.exists():
        with open('train_data.csv', mode='w', newline='') as csvfile:
            writer_object = writer(csvfile)
            writer_object.writerow(column_names)
            csvfile.close()
    data_arr = np.zeros((1, len(column_names[:-1])))
    curr_count = 0
    # load and resize video, keeping aspect ratio constant
    rawCap = cv2.VideoCapture(f"./working_files/{args.test_file}.mp4")

    initRet, initFrame = rawCap.read()
    h, w = initFrame.shape[:2]
    h_new = 360
    w_new = int(w*(h_new/h))
    dim = (w_new, h_new)
    fps = rawCap.get(cv2.CAP_PROP_FPS)
    print(f'Frame dimensions: {dim}, and FPS: {fps}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    resized = cv2.VideoWriter("./working_files/resized_vid.mp4", fourcc, fps, dim)

    while True:
        initRet, initFrame = rawCap.read()
        if initRet==True:
            newFrame = cv2.resize(initFrame, dim, interpolation = cv2.INTER_AREA)
            resized.write(newFrame)
        else:
            break

    rawCap.release()
    resized.release()

    cap = cv2.VideoCapture("./working_files/resized_vid.mp4")
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while cap.isOpened():
        
        ret, frame = cap.read()

        if ret:
            pass
        else:
            break
        
        result = DeepFace.analyze(frame,actions = ['emotion'], enforce_detection=False, prog_bar=False)

        # Read in all the emotions' intensity values
        curr_arr = [result['emotion']['happy'], result['emotion']['angry'], result['emotion']['disgust'], result['emotion']['sad'], result['emotion']['fear'], result['emotion']['neutral'], result['emotion']['surprise']]
        data_arr += np.array(curr_arr)# Cumulative sum throughout video
        curr_count += 1 # keep track of current frame count and append data at fixed intervals
        if bool(strtobool(args.test)) is not True:
            if (curr_count % args.seq_len)==0:
                curr_count = 0
                data_arr = data_arr/np.sum(data_arr[0]) # to 'scale' values to their proportion of total
                data_arr = np.concatenate(([[False if f'{args.test_file}'=='not_depressed' else True]], data_arr), axis=1)
                # Append values to csv file
                with open('train_data.csv', 'a', newline='') as csvfile:
                    writer_object = writer(csvfile)
                    writer_object.writerow(data_arr[0])     
                    csvfile.close()
                data_arr = np.zeros((1,len(curr_arr)))
        else:
            pass # if test argument is true, pass to only output one observation (i.e. the test.mp4 vid)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.1,4)

        # Draw rectangle around face
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        font = cv2. FONT_HERSHEY_SIMPLEX

        # Dominant emotion shown
        cv2.putText(frame,
                    result['dominant_emotion'],
                    (x,y-10),
                    font, 3,
                    (0, 0, 255),
                    2,
                    cv2.LINE_4)
        cv2.imshow('Original video', frame)

        # 'q' key to interrupt before video ends
        if cv2.waitKey(2) & 0xFF ==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 
    if bool(strtobool(args.test)) is True:
        data_arr = data_arr/np.sum(data_arr[0]) 
        np.save("test.npy", data_arr[0])
        
    if bool(strtobool(args.audio)) is True: # run and add audio analyse if argument is True
        audio_df = audio_analyse(str(args.test_file))
        emotion_data = pd.read_csv("train_data.csv")
        emotion_df = pd.DataFrame(emotion_data)

        data_horizontal = pd.concat([emotion_df, audio_df], axis=1) 

        data_horizontal.to_csv("train_data.csv", header=True) # add to training data
    

if __name__ == '__main__':
    # Arguments for main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_file',
        choices=['depressed', 'not_depressed'],
        default='depressed',
        type=str)
    parser.add_argument(
        '--seq_len',
        default=20,
        type=int)
    parser.add_argument(
        '--audio',
        default='False',
        choices=['True', 'False'])
    parser.add_argument(
        '--test',
        default='False',
        choices=['True', 'False'])
    
    args = parser.parse_args()
    main(args)