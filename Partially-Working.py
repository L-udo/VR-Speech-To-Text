import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import threading
import queue
import katosc

kat =  katosc.KatOsc()
SR = 44100                    #SampleRate
Audio_chunk_duration = 3        #duration in seconds
device = 5                      #Device as listed in print(sd.query_devices())
KATOSCSPEED = 16 #default 6
q = queue.Queue(maxsize=3)


def rec_audio():
    print("rec_audio() started")
    global device
    global Audio_chunk_duration
    global SR
    while True:
        sd.default.device = device
        record = sd.rec(int(Audio_chunk_duration * SR), samplerate=SR, channels=2)
        sd.wait()
        q.put(record)
        print("Recorded")

def transcribe_Audio():
    print("transcribe_Audio() started")
    model = whisper.load_model("small")
    while True:
        if q.empty() != True:
            write('output.wav', SR, q.get())
            print("Writing file done")
            result = model.transcribe("output.wav")
            print(result["text"]) #placeholder for KATOSC
            kat.set_text(result["text"])
        else:
            pass


     
threading.Thread(target=transcribe_Audio, daemon=False).start()
threading.Thread(target=rec_audio, daemon=True).start()

