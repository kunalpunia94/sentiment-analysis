# Multimodal Multilingual Sentiment Intelligence Platform

This project is a Streamlit application that provides a unified pipeline for sentiment analysis across text, speech, audio, and video.

## Project Structure

```
.
├── .gitignore
├── app.py
├── assets
│   └── img.png
├── packages.txt
├── requirements.txt
├── services
│   ├── language_detect.py
│   ├── sentiment.py
│   ├── speech_to_text.py
│   ├── translation.py
│   └── video_audio.py
└── utils.py
```

## How to Run

1.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
