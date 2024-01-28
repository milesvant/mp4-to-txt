# mp4-to-txt

This tool is designed to automate the process of transcribing audio from video files. It extracts audio from `.mp4` videos, transcribes them using OpenAI's Whisper API, and saves the transcriptions as `.txt` files.

## Prerequisites

- Python 3.6 or higher.
- [OpenAI API key](https://platform.openai.com/docs/quickstart/account-setup) 

## Setup

- Place your `.mp4` video files in the `videos` directory.

## Usage

- Run the script using the command:
    ```bash
    OPENAI_API_KEY=[YOUR KEY HERE] python transcriber.py
    ```
