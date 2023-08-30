import json
from src.data_models import ClipData


with open('./episode-15-data/clips_data.json', 'r') as f:
    data = json.loads(f.read())

data = ['\n'.join(ClipData.from_pytree(item).audio_transcriptions_raw) for item in data]

with open('./episode-15-transcriptions.json', 'r') as f:
    better_data = json.loads(f.read())

assert len(data) == len(better_data)

for transcription, better_transcription in zip(data, better_data):
    print(transcription)
    print(better_transcription)
    print()
