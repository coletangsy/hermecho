
# Hermecho

> A fusion of "Hermes," the Greek messenger god symbolizing interpretation and communication, with "Echo," the nymph who repeats spoken words. This evokes the tool's transcription of Korean audio and its translation into Chinese subtitles, bridging languages like a divine echo.

This project is a high-performance, command-line tool for translating videos with Korean audio into videos with Traditional Chinese (Taiwan) subtitles. It leverages a sophisticated pipeline that includes local transcription, intelligent LLM-based translation, and automated subtitle generation.

## Key Features

*   **Automated End-to-End Pipeline:** From video input to a final, subtitled video output with minimal user intervention.
*   **High-Quality Transcription:** Utilizes a local instance of OpenAI's **Whisper** model for accurate speech-to-text with timestamps.
*   **Intelligent, Context-Aware Translation:** Employs a Large Language Model (e.g., Gemini 2.5 Pro) with a highly optimized prompt that performs several advanced tasks:
    *   **Contextual Correction:** Uses a reference file to intelligently correct transcription errors in names and key terminology.
    *   **Preservation of Names & Terms:** Ensures that Korean names and specified English words are preserved in their original form, preventing incorrect transliteration or translation.
*   **Optimized for Performance:** The translation process is architected for speed and efficiency:
    *   **Large Batch Processing:** Takes full advantage of the large context windows in modern LLMs to process up to 200 segments at once.
    *   **Concurrent Fallback:** In case of an error, the system automatically falls back to a concurrent, segment-by-segment translation to ensure robustness without sacrificing speed.
*   **Robust Error & Gap Handling:** The pipeline includes several guardrails for a more professional result:
    *   **Transcription Gap Filling:** Automatically detects and fills long periods of silence with a `[no speech]` placeholder.
    *   **Strict JSON Communication:** Enforces a strict JSON-based workflow with the LLM to prevent malformed outputs and conversational filler.
*   **Subtitle Generation:** Creates a standard `.srt` subtitle file and burns it directly into the final video.

## Demo
[![Watch the video](https://raw.githubusercontent.com/coletangsy/hermecho/main/demo/yooyeon_250114.png)](https://raw.githubusercontent.com/coletangsy/hermecho/main/demo/yooyeon_250114_translated.mp4)

[[Link to Original Video](https://www.youtube.com/watch?v=cWexFmUagsc)] 

*This video demonstrates the final output of the solution, featuring subtitles created by this project. (video compressed for uploading)*



## How it Works

The video translation process is a multi-stage pipeline designed for quality and robustness:

1.  **Audio Extraction:** The audio is extracted from the input video using `ffmpeg`.
2.  **Transcription:** The Korean audio is transcribed into time-coded text segments using a local **Whisper** model.
3.  **Gap Filling:** The transcription is scanned for significant time gaps, which are filled with a placeholder.
4.  **Intelligent Translation:** The text segments are sent to an LLM for translation. The system automatically chooses the best strategy:
    *   For short videos, the entire text is translated in a single, efficient batch.
    *   For long videos, the text is translated in large, overlapping chunks to maintain context and maximize speed.
5.  **Subtitle Generation:** The translated and corrected text segments are used to generate a final `.srt` subtitle file.
6.  **Video Finalization:** A new video file is created with the generated subtitles burned directly into it.

## Getting Started

### Prerequisites

*   Python 3.9+
*   `ffmpeg`: This must be installed on your system. On macOS, you can use Homebrew: `brew install ffmpeg`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/coletangsy/hermecho.git
    cd hermecho
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` file needs to be created or updated. Key dependencies include `openai-whisper`, `langchain`, `langchain-openai`, `pydub`, `noisereduce`, `tqdm`, and `python-dotenv`.*

3.  **Create a `.env` file** in the root directory to store your API key:
    ```
    OPENROUTER_API_KEY="your_openrouter_api_key"
    ```

## Usage

To translate a video, run the `main.py` script from the `src` directory:

```bash
python src/main.py my_video.mp4
```

### Command-Line Arguments

You can customize the process using several optional arguments:

*   `--model`: The Whisper model to use for transcription (e.g., `base`, `small`, `medium`, `large`). Default is `large`.
*   `--language`: The language of the audio. Default is `ko`.
*   `--target_language`: The target language for translation. Default is `Traditional Chinese (Taiwan)`.
*   `--reference_file`: Path to a reference file (e.g., a list of names) to improve translation accuracy.

For more information, run:

```bash
python src/main.py --help
```


## Project Structure

The project is organized into a modular structure to separate concerns and improve maintainability.

```
/
├── .env                  # Stores API keys and other secrets.
├── .gitignore            # Specifies files for Git to ignore.
├── DESIGN.md             # The detailed technical design document for the project.
├── README.md             # This file.
├── requirements.txt      # Lists all Python dependencies for easy installation.
│
├── input/                # Directory for placing your source video files.
│
├── output/               # Directory where all generated files are saved.
│   └── {video_name}/     # Each video gets its own subdirectory.
│       ├── subtitles.srt
│       └── translated.mp4
│
├── references/           # Directory for context-aware translation files.
│   └── tripleS.md        # Example reference file containing names or terms.
│
└── src/                  # Contains all the core application logic.
    ├── main.py           # Main entry point and orchestrator of the translation pipeline.
    ├── video_processing.py # Handles all direct video/audio manipulation using ffmpeg.
    ├── transcription.py  # Manages the audio-to-text process using the Whisper model.
    ├── translation.py    # Core translation logic, including prompt engineering and LLM communication.
    ├── subtitles.py      # Logic for generating, cleaning, and adjusting subtitle timings.
    └── utils.py          # Utility functions, such as loading reference files.

```
