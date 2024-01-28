import concurrent.futures
import os
import time

from pathlib import Path

import openai
import requests

from moviepy.editor import VideoFileClip
from tqdm import tqdm
from pydub import AudioSegment


AUDIO_PATH = Path('./audio')
VIDEO_PATH = Path('./videos')
TRANSCRIPTS_PATH = Path('./transcripts')

WHISPER_COST_PER_SECOND = 0.0001

CHUNK_LENGTH_MS = 5 * 60 * 1000  # 5 minutes.


def get_untranscribed_video_files():
    all_video_files = [file for file in os.listdir(VIDEO_PATH) if file.endswith('.mp4')]
    all_transcript_files = [file for file in os.listdir(TRANSCRIPTS_PATH) if file.endswith('.txt')]

    untranscribed_video_files = []
    for video_file in all_video_files:
        transcript_file = video_file.replace('.mp4', '.txt')
        if transcript_file not in all_transcript_files:
            untranscribed_video_files.append(video_file)

    return untranscribed_video_files


def list_mp4s_and_total_duration(mp4_files) -> bool:
    total_duration = 0
    max_filename_length = max([len(file) for file in mp4_files])

    print("Video files found: ")
    for file in mp4_files:
        filepath = VIDEO_PATH / file
        with VideoFileClip(str(filepath)) as video:
            duration = video.duration
            total_duration += duration
            print(f"* {file} {'-' * (1 + max_filename_length - len(file))} {duration:.2f} seconds")

    print(f"\nCumulative Runtime: {total_duration:.2f} seconds")
    print(f"Estimated transcription cost: ${total_duration * WHISPER_COST_PER_SECOND:.2f}")

    # Print without newline.
    print("Do you want to continue? [y/N] ", end='')
    response = input()
    if response.lower() == 'y':
        return True
    else:
        return False


def extract_audio(video_file):
    video = VideoFileClip(str(VIDEO_PATH / video_file))
    audio_filename = video_file.replace('.mp4', '.mp3')
    video.audio.write_audiofile(str(AUDIO_PATH / audio_filename), logger=None)  # Disable logging
    audio = AudioSegment.from_mp3(str(AUDIO_PATH / audio_filename))

    # Splitting the audio
    chunks = [audio[i:i + CHUNK_LENGTH_MS] for i in range(0, len(audio), CHUNK_LENGTH_MS)]
    chunk_filenames = []
    for idx, chunk in enumerate(chunks):
        chunk_filename = f"{audio_filename.replace('.mp3', '')}_{idx}.mp3"
        chunk.export(AUDIO_PATH / chunk_filename, format="mp3")
        chunk_filenames.append(chunk_filename)

    return chunk_filenames


def transcribe_audio(client, audio_filename):
    response = client.audio.transcriptions.create(
        file=open(AUDIO_PATH / audio_filename, "rb"),
        model="whisper-1"
    )
    return response.text


def combine_transcripts(transcripts, original_filename):
    combined_transcript = "\n".join(transcripts)
    filename = Path(original_filename).name.replace('.mp3', '.txt')
    with open(TRANSCRIPTS_PATH / filename, 'w') as f:
        f.write(combined_transcript)


def transcribe_audio(client, audio_filename):
    max_retries = 5
    retry_delay = 10  # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            response = client.audio.transcriptions.create(
                file=open(AUDIO_PATH / audio_filename, "rb"),
                model="whisper-1"
            )
            return response.text

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limit error
                print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise e  # Re-raise the exception if it's not a rate limit error

        except Exception as e:
            raise e  # Re-raise for any other exceptions

    raise RuntimeError(f"Failed to transcribe after {max_retries} attempts.")


def main():
    video_files = get_untranscribed_video_files()

    if len(video_files) == 0:
        print("No video files found to transcribe. Either all videos have been transcribed or no videos have been downloaded.")
        return

    proceed = list_mp4s_and_total_duration(video_files)
    if not proceed:
        return

    with concurrent.futures.ThreadPoolExecutor() as executor:
        audio_progress = tqdm(total=len(video_files), desc="Audio Extraction", unit="file")
        transcribe_progress = tqdm(desc="Transcription", unit="chunk")

        client = openai.OpenAI()

        for video_file in video_files:
            # Extract audio and split into chunks
            chunk_filenames = extract_audio(video_file)
            audio_progress.update(1)

            # Transcribe each chunk
            futures = [executor.submit(transcribe_audio, client, chunk_filename) for chunk_filename in chunk_filenames]
            transcripts = [future.result() for future in concurrent.futures.as_completed(futures)]
            transcribe_progress.update(len(chunk_filenames))

            # Combine transcripts
            combine_transcripts(transcripts, video_file.replace('.mp4', '.mp3'))

        audio_progress.close()
        transcribe_progress.close()


if __name__ == '__main__':
    main()
    print("Exiting.")

