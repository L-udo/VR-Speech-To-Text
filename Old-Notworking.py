import sounddevice as sd
import numpy as np
import whisper
import asyncio
import sys
import katosc
import dotenv
from textwrap import wrap
import time
import threading

dotenv_file = "SETTINGS.env"
kat =  katosc.KatOsc()
wrapped_message = [""]

#SETTINGS
KATOSCSPEED = 6 #default 6
#Time in seconds each group of 128 characters is show
CHUNKSIZE = 128 #default 128
#Length of each chunk of text sent to kat, Chane if you have increased the number of characters on avatar
MODEL_TYPE="base.en"
# the model used for transcription. https://github.com/openai/whisper#available-models-and-languages
LANGUAGE="English"
# pre-set the language to avoid autodetection
HW_DEVICE="cuda"
# the hardwared used to run whisper cpu/cuda
BLOCKSIZE= 16000 
# this is the base chunk size the audio is split into in samples. blocksize / 16000 = chunk length in seconds. 
SILENCE_THRESHOLD=800 #default 400
# should be set to the lowest sample amplitude that the speech in the audio material has
SILENCE_RATIO=100
# number of samples in one buffer that are allowed to be higher than threshold
MICROPHONE=5


global_ndarray = None
model = whisper.load_model(MODEL_TYPE, device = HW_DEVICE)

async def inputstream_generator():
	"""Generator that yields blocks of input data as NumPy arrays."""
	q_in = asyncio.Queue()
	loop = asyncio.get_event_loop()

	def callback(indata, frame_count, time_info, status):
		loop.call_soon_threadsafe(q_in.put_nowait, (indata.copy(), status))

	stream = sd.InputStream(samplerate=16000, channels=1, dtype='int16', blocksize=BLOCKSIZE, callback=callback, device=MICROPHONE)
	with stream:
		while True:
			indata, status = await q_in.get()
			yield indata, status
			
		
async def process_audio_buffer():
	global global_ndarray
	async for indata, status in inputstream_generator():
		
		indata_flattened = abs(indata.flatten())
				
		# discard buffers that contain mostly silence
		if(np.asarray(np.where(indata_flattened > SILENCE_THRESHOLD)).size < SILENCE_RATIO):
			continue
		
		if (global_ndarray is not None):
			global_ndarray = np.concatenate((global_ndarray, indata), dtype='int16')
		else:
			global_ndarray = indata
			
		# concatenate buffers if the end of the current buffer is not silent
		if (np.average((indata_flattened[-100:-1])) > SILENCE_THRESHOLD/15):
			continue
		else:
			local_ndarray = global_ndarray.copy()
			global_ndarray = None
			indata_transformed = local_ndarray.flatten().astype(np.float32) / 32768.0
			result = model.transcribe(indata_transformed, language=LANGUAGE)
			wrapped_message.append(wrap(result["text"], 128))
			
		del local_ndarray
		del indata_flattened


async def main():
	settings()
	print('\nWhisper is Listening ...\n')
	threading.Thread(target= slowKATosc, daemon=True).start()
	audio_task = asyncio.create_task(process_audio_buffer())
	while True:
		await asyncio.sleep(1)
	audio_task.cancel()
	try:
		await audio_task
	except asyncio.CancelledError:
		print('\nwire was cancelled')


def settings():
	if dotenv.get_key(dotenv_file,"DEFAULT") == "1":
		print(sd.query_devices())
		print("\n")
		print("It looks like you are running with default settings\n please set the microphone you want to use")
		print("Microphone_NUM:")
		dotenv.set_key(dotenv_file,"MICROPHONE", input())
		print("Settings Saved!",flush=True)
		dotenv.set_key(dotenv_file,"DEFAULT", "0")
	else:
		pass


def slowKATosc():
	import time
	global wrapped_message
	while True:
		try:
			print(wrapped_message[0])
			print(len(wrapped_message))
			katmsg = wrapped_message[0]
			kat.set_text(katmsg)	
			wrapped_message.pop(0)
		except:
			print("bingus")
			print(len(wrapped_message))
		time.sleep(2)




if __name__ == "__main__":
	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		sys.exit('\nInterrupted by user')