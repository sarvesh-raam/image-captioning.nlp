FROM python:3.12-slim

# Hugging Face Spaces runs containers as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Only the source files are copied, so a stale checkpoint in the Space repo can never shadow
# the one on the Model Hub.
COPY --chown=user api.py inference.py model.py data_loader.py ./

# Bake the model (and the CLIP backbone, if the model uses one) into the image at build time
# so cold starts don't re-download anything. Fails the build if the model can't be loaded.
RUN python -c "import api; api.load_model(); assert api.generator is not None, api.load_error"

EXPOSE 7860
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
