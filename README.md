# AI Audio Dataset Creator for TTS/RVC

This project provides a set of Python scripts and a Streamlit UI to automate the process of creating audio datasets from video files, suitable for training Text-to-Speech (TTS) or Retrieval-based Voice Conversion (RVC) models.

The pipeline includes:
1.  **Audio Extraction:** Extracts audio from video files.
2.  **Vocal Isolation (Optional):** Uses Demucs to remove background noise and music, isolating vocals.
3.  **Transcription & Alignment:** Utilizes WhisperX (with a model as whisper large-v3-turbo) for accurate transcription and word-level timestamp alignment in Hungarian (or other configured languages).
4.  **Speaker Diarization (Optional):** Identifies and separates different speakers in the audio.
5.  **Segmentation:** Splits the audio into short segments (configurable length, e.g., 3-13 seconds, with a loose upper limit) based on sentences or natural pauses, ensuring segments contain speech from a single speaker (if diarization is enabled).
6.  **Metadata Generation:** Creates a `metadata.csv` file linking audio segments to their transcriptions, speaker IDs, confidence scores, and a flag for manual review.

The output is a structured dataset ready to be used as input for scripts that prepare data for specific TTS model formats (columns: file_path,transcription,duration_ms,speaker_id,confidence,needs_review).

## Features

*   Process single or multiple video files, or an entire folder of videos.
*   User-friendly Streamlit interface for configuration and process monitoring.
*   Optional background noise removal using Demucs.
*   Optional speaker diarization using `pyannote.audio` through WhisperX.
*   Configurable segment length and transcription confidence threshold.
*   Supports GPU (CUDA) for faster processing, with automatic fallback to CPU.
*   Generates a `metadata.csv` for easy integration with further processing scripts.
*   Organizes outputs into dataset-specific and video-specific subfolders.

## Prerequisites

*   **Python 3.10+** (Python 3.11 recommended)
*   **Conda** (recommended for environment management) or `venv`
*   **FFmpeg:** Must be installed system-wide and accessible from your PATH. Download from [ffmpeg.org](https://ffmpeg.org/download.html).
*   **CUDA-enabled GPU (Optional but Highly Recommended):** For significantly faster processing with Demucs and WhisperX. Ensure your NVIDIA drivers and CUDA toolkit are correctly installed.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/csokosgeza/ai-audio-dataset-creator.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a Conda environment (recommended):**
    ```bash
    conda create --name dataset_env python=3.11 -y
    conda activate dataset_env
    ```
    Alternatively, use `venv`:
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # Linux/macOS: source venv/bin/activate
    ```

3.  **Install PyTorch:**
    Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select the appropriate options for your system (OS, Package: Conda/Pip, Language: Python, Compute Platform: CUDA version or CPU). Run the provided installation command.
    *Example for CUDA 11.8 with Conda:*
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    ```
    *Example for CPU only with Conda:*
    ```bash
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
    ```

4.  **Install WhisperX from GitHub:**
    ```bash
    pip install git+https://github.com/m-bain/whisperX.git@main
    ```
    *(For GPU, ensure PyTorch with CUDA is installed first. WhisperX might have a `[gpu]` extra, but often relies on the PyTorch installation.)*

5.  **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Speaker Diarization Setup (If `use_diarization: True`):**
    *   You will need a Hugging Face account and an Access Token (with at least 'read' permissions).
    *   Go to [Hugging Face Tokens](https://huggingface.co/settings/tokens) to create a token.
    *   **Crucially, you must accept the user agreements for the following `pyannote.audio` models on Hugging Face while logged in:**
        *   [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
        *   [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)
        *   [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) (or `pyannote/speaker-diarization-3.1` if the former gives issues)
    *   Copy your Hugging Face token.

## Configuration

1.  Rename `config.example.yaml` to `config.yaml`.
2.  Edit `config.yaml` with your desired settings:
    *   `input_video`: Path to your video file or a folder containing video files.
    *   `output_base_dir`: The main directory where your datasets will be saved.
    *   `language`: Set to `"hu"` for Hungarian.
    *   `use_diarization`: Set to `True` if you want to separate speakers.
    *   `huggingface_token`: If `use_diarization` is `True`, paste your Hugging Face Access Token here.
    *   Adjust other parameters like `target_sample_rate`, `whisper_model_size`, segment durations, etc., as needed.

## Usage

1.  Ensure your Conda (or venv) environment is activated.
2.  Navigate to the project directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your web browser.
5.  **Using the UI:**
    *   **Dataset Mode:**
        *   Choose "Új adathalmaz létrehozása" (Create new dataset) and provide a name.
        *   Or, choose "Meglévő adathalmazhoz adás" (Add to existing dataset) and select a previously created dataset folder.
    *   **Input:** Upload video file(s) or provide a path to a single video or a folder of videos.
    *   **Configure Settings:** Adjust parameters in the sidebar as needed. You can save your current UI settings to `config.yaml` for future use.
    *   **Start Processing:** Click the "Teljes Feldolgozás Indítása" (Start Full Processing) button.
    *   Monitor the progress in the main panel.
6.  **Output:**
    *   Processed data will be saved in `output_base_dir/YOUR_DATASET_NAME/VIDEO_NAME/`.
    *   Each `VIDEO_NAME` subfolder will contain:
        *   `audio_raw.wav` (extracted audio)
        *   `audio_clean.wav` (if Demucs was used)
        *   `segments/` (or `segments/SPEAKER_ID/` if diarization was used) containing the `.wav` audio segments.
    *   A **common `metadata.csv`** file will be created/updated in the `output_base_dir/YOUR_DATASET_NAME/` directory, containing paths (relative to this dataset directory) and transcriptions for all processed segments from all videos in that dataset.

## Notes

*   Processing, especially with Demucs and `large-v3` Whisper model on CPU, can be very time-consuming. Using a CUDA-enabled GPU is highly recommended.
*   The quality of the dataset heavily depends on the quality of the input audio and the accuracy of the Whisper transcription. Manual review and correction of the `metadata.csv` (especially for low-confidence segments) is advised for best results.
*   The generated `metadata.csv` and audio segments are intended as an intermediate dataset. You will need a separate script to convert this into the final `input_ids`, `labels`, and `attention_mask` format required by your specific TTS model (e.g., using an LLaMA tokenizer).

## TODO / Future Improvements

*   Implement a more robust way to merge/append to the `metadata.csv` if the Streamlit app is run multiple times for the same dataset.
*   Add a UI section for reviewing and editing transcripts directly within the Streamlit app.
*   More granular error handling and user feedback during long processes.
*   Option to split overly long Whisper segments more intelligently.

## Support the Project

If you find this project useful and would like to support its development, you can do so through:

*   **GitHub Sponsors:** []
*   **Buy Me a Coffee:** [![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-ffdd00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/csokosgeza)

Your support is greatly appreciated!
