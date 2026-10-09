FROM python:3.12-slim
WORKDIR /app
# WITH_ASR=1 adds faster-whisper for the /transcribe endpoint (the Whisper model downloads on first use)
ARG WITH_ASR=0
COPY requirements.txt requirements-asr.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
 && if [ "$WITH_ASR" = "1" ]; then pip install --no-cache-dir -r requirements-asr.txt; fi
COPY puente/ puente/
COPY serve/ serve/
COPY public/ public/
# gemini (set GEMINI_API_KEY at run time), or rules for the keyword baseline with no key
ENV PUENTE_BACKEND=gemini
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
