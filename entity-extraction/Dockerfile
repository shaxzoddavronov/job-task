# Plan section 10. The app container does extraction and OCR; the model runs in
# the separate Ollama container, which is the only thing that needs a GPU.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Docling's layout models are baked in below, so nothing is fetched while a
    # document is being analysed.
    HF_HOME=/opt/models \
    DOCLING_ARTIFACTS_PATH=/opt/models/docling

# Tesseract plus the two Uzbek language packs. tesseract-ocr-uzb covers Latin
# and tesseract-ocr-uzb-cyrl covers Cyrillic; Docling is configured to try both.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr \
      tesseract-ocr-uzb \
      tesseract-ocr-uzb-cyrl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Torch arrives as a Docling dependency, and by default pip takes the CUDA
# build: 1.2 GB of torch plus 3.2 GB of nvidia/* and 0.9 GB of triton that this
# container can never use, because only Ollama is given a GPU. That is the
# difference between a 12.7 GB image and a 5 GB one.
#
# The CPU wheel is installed first, pinned to the *same* version Docling
# resolves to and carrying the +cpu local tag. Installing an older version
# instead just gets it upgraded back to the CUDA build when Docling resolves.
# torchvision must come from the same index: the PyPI build is compiled against
# CUDA torch and dies with "operator torchvision::nms does not exist", which
# surfaces only when Docling loads its layout model — the image builds fine.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      "torch==2.14.1+cpu" \
      "torchvision==0.29.1+cpu"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import torch; assert torch.version.cuda is None, torch.__version__; print('torch', torch.__version__)"

# Bake the layout and table models into the image at build time (plan section
# 4). Only these two are used: the defaults would also fetch a code/formula
# model, a picture classifier and RapidOCR, none of which this pipeline calls.
#
# output_dir has to match DOCLING_ARTIFACTS_PATH above. Without it the weights
# land in the Hugging Face cache layout instead and Docling rejects the
# artifacts path at request time — the image builds fine and fails on the first
# upload.
RUN python - <<'DOCLING'
from pathlib import Path

from docling.utils.model_downloader import download_models

download_models(
    output_dir=Path("/opt/models/docling"),
    with_layout=True,
    with_tableformer=True,
    with_code_formula=False,
    with_picture_classifier=False,
    with_rapidocr=False,
    progress=False,
)
DOCLING

# Nothing may be downloaded while a document is being analysed.
ENV HF_HUB_OFFLINE=1

COPY app ./app

# Convert a generated PDF at build time. Checking versions is not enough: a
# mismatched torchvision passes every import check and then fails when Docling
# loads its layout model, so the only honest test is a real conversion.
RUN python - <<'SMOKE'
import pymupdf

from app.extract_text import extract

doc = pymupdf.open()
page = doc.new_page()
page.insert_text((60, 70), "1. Vazirlik hisobot taqdim etsin, 1-noyabrgacha.", fontsize=11)
doc.save("/tmp/smoke.pdf")
doc.close()

result = extract("/tmp/smoke.pdf", ocr=False)
assert result.blocks, "Docling produced no blocks"
assert "Vazirlik" in result.text, result.text
print("build smoke test OK:", result.text[:60])
SMOKE

EXPOSE 8000

# The app answers /health as soon as it is up, and reports there whether the
# model is ready yet.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
