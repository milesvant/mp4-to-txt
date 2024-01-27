from pathlib import Path

import openai
from moviepy.editor import VideoFileClip
import concurrent.futures
import os


AUDIO_PATH = Path('./audio')
VIDEO_PATH = Path('./videos')
TRANSCRIPTS_PATH = Path('./transcripts')


def extract_audio(video_file):
    print(f"Extracting audio from video: {video_file}\n")
    video = VideoFileClip(str(VIDEO_PATH / video_file))
    audio_filename = video_file.replace('.mp4', '.mp3')
    video.audio.write_audiofile(str(AUDIO_PATH / audio_filename))
    return audio_filename


def transcribe_audio(audio_filename):
    client = openai.OpenAI()
    print(f"\nTranscribing audio: {audio_filename} with OpenAI API...\n")
    response = client.audio.transcriptions.create(
        file=open(AUDIO_PATH / audio_filename, "rb"),
        model="whisper-1"
    )
    transcript = response.text

    print(f"Generated transcript for {audio_filename}: {transcript}\n")
    filename = Path(audio_filename).name.replace('.mp3', '.txt')
    with open(TRANSCRIPTS_PATH / filename, 'w') as f:
        f.write(transcript)


def main():
    video_files = [file for file in os.listdir(VIDEO_PATH) if file.endswith('.mp4')]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_audio = {executor.submit(extract_audio, video_file): video_file for video_file in video_files}

        for future in concurrent.futures.as_completed(future_audio):
            audio_filename = future.result()
            executor.submit(transcribe_audio, audio_filename)


if __name__ == '__main__':
    main()

