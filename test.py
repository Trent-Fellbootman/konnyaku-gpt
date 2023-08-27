from src.data_models import ClipData, ClipsMetadata


a = ClipData(video_path='test', audio_transcription_raw='test', screenshot_description='test')

print(a.to_json())

assert a == ClipData.from_json(a.to_json())

b = ClipsMetadata(
    ['1.mp4', '2.mp4', '4.mp4']
)

print(b.to_json())

assert b == ClipsMetadata.from_json(b.to_json())
