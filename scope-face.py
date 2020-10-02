import pyaudio
import numpy as np
import cv2

def main():
    fs = 44100
    p = pyaudio.PyAudio()
    channels = 2

    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=fs,
                    output=True)

    cap = cv2.VideoCapture(4)

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',gray)
        blur = cv2.GaussianBlur(gray,(11,11),0)
        cv2.imshow('blur',blur)
        ret, th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('bin',th)
        contours, hierarchy = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        edges = np.zeros(gray.shape)
        cv2.drawContours(edges, contours, -1, 255, 3)
        cv2.imshow('contour',edges)
        contours = np.concatenate(contours).astype('float32').squeeze()
        contours[:, 0] = contours[:, 0] - gray.shape[0]/2
        contours[:, 1] = gray.shape[1]/2 - contours[:, 1]
        contours /= max(gray.shape)
        contours = np.tile(contours, (100, 1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        stream.write(contours)

    # When everything done, release the capture
    cap.release()
    
    return contours

    stream.stop_stream()
    stream.close()

    p.terminate()


if __name__ == "__main__":
    main()