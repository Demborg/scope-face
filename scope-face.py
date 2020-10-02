import pyaudio
import numpy as np
import cv2

def _make_heart(t: np.ndarray) -> np.ndarray:
    f = 440.0        # sine frequency, Hz, may be float
    data = np.stack(
        [
            16 * np.sin(f * t) ** 3,
            13 * np.cos(f * t) - 5 * np.cos(2 * f * t)  - 2 *np.cos(3 * f * t) - np.cos(4 * f * t)
        ],
        axis=-1
    )
    data = (1 + 0.1 * np.sin(t))[:, np.newaxis] * data
    data /= data.max()
    data = data.reshape(-1)
    return data
    
def _get_picture_contour():
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    
    contours = np.concatenate(contours).astype('float32').squeeze()
    contours[:, 0] = gray.shape[0]/2 - contours[:, 0]
    contours[:, 1] = gray.shape[1]/2 - contours[:, 1]
    contours /= max(gray.shape)
    contours = np.tile(contours, (10000, 1))
    return contours

def main():


    p = pyaudio.PyAudio()
    fs = 44100       # sampling rate, Hz, must be integer
    duration = 10   # in seconds, may be float
    channels = 2

    # generate samples, note conversion to float32 array
    t = 2*np.pi*np.arange(fs*duration)/fs
    samples = (_get_picture_contour()).astype(np.float32)
    # samples = (_make_heart(t)).astype(np.float32)
    print(len(samples))

    stream = p.open(format=pyaudio.paFloat32,
                    channels=channels,
                    rate=fs,
                    output=True)

    stream.write(samples)

    stream.stop_stream()
    stream.close()

    p.terminate()


if __name__ == "__main__":
    main()