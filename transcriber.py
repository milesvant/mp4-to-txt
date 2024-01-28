from pathlib import Path
import openai
from moviepy.editor import VideoFileClip
import concurrent.futures
import os
from tqdm import tqdm


AUDIO_PATH = Path('./audio')
VIDEO_PATH = Path('./videos')
TRANSCRIPTS_PATH = Path('./transcripts')

WHISPER_COST_PER_SECOND = 0.0001


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
    return audio_filename


def transcribe_audio(client, audio_filename):
    response = client.audio.transcriptions.create(
        file=open(AUDIO_PATH / audio_filename, "rb"),
        model="whisper-1"
    )
    transcript = response.text

    filename = Path(audio_filename).name.replace('.mp3', '.txt')
    with open(TRANSCRIPTS_PATH / filename, 'w') as f:
        f.write(transcript)


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
        transcribe_progress = tqdm(total=len(video_files), desc="Transcription", unit="file")

        # Extract audio
        future_audio = [executor.submit(extract_audio, video_file) for video_file in video_files]
        for future in concurrent.futures.as_completed(future_audio):
            future.result()  # Wait for each audio extraction to complete
            audio_progress.update(1)

        # Transcribe audio
        client = openai.OpenAI()
        future_transcribe = [executor.submit(transcribe_audio, client, audio_filename.replace('.mp4', '.mp3')) for audio_filename in video_files]
        for future in concurrent.futures.as_completed(future_transcribe):
            future.result()  # Wait for each transcription to complete
            transcribe_progress.update(1)

        audio_progress.close()
        transcribe_progress.close()

if __name__ == '__main__':
    main()
    print("Exiting.")

