# server.py

import os
import numpy as np
import re
import json
import io
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import pytesseract
import easyocr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Initialise EasyOCR pour le français, anglais, italien et espagnol
reader = easyocr.Reader(['fr', 'en', 'it', 'es'], gpu=False)

app = FastAPI()


class FicheVinResponse(BaseModel):
    ocrTexte: str
    nom: str = ""
    domaine: str = ""
    appellation: str = ""
    millesime: str = ""
    region: str = ""
    pays: str = ""
    couleur: str = ""
    cepage: str = ""
    degreAlcool: str = ""
    prixEstime: float = 0.0
    imageEtiquetteUrl: str = ""
    tempsGarde: str = ""


def ameliorer_image(image: Image.Image) -> Image.Image:
    """
    Améliore le contraste et la netteté pour l'OCR.
    """
    import cv2

    # Conversion PIL → OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Upscale pour meilleure reconnaissance
    img_cv = cv2.resize(img_cv, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Binarisation adaptative
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 10
    )
    return Image.fromarray(thresh)


def nettoyer_texte_ocr(texte: str) -> str:
    """
    Nettoie les caractères parasites de l'OCR.
    """
    texte = re.sub(r'[^\wÀ-ÿ%.€\s-]', '', texte)
    return re.sub(r'\s+', ' ', texte).strip()


def demander_infos_gpt(texte_ocr: str) -> dict:
    """
    Envoie un prompt à l'API OpenAI pour extraire et enrichir les champs vin
    (cépage, degré, prix, temps de garde).
    """
    prompt = f"""
Tu es un assistant expert en vins.  
Reçois ce texte OCR :
{texte_ocr}

1) Extrait sans recherches :
   - nom, domaine, appellation, millesime, region, pays, couleur  
2) Fais ensuite une vraie recherche web pour trouver :
   - cepage
   - degreAlcool
   - prixEstime
   - tempsGarde

Retourne uniquement un JSON valide, sans autre texte :
{{
  "nom": "",
  "domaine": "",
  "appellation": "",
  "millesime": "",
  "region": "",
  "pays": "",
  "couleur": "",
  "cepage": "",
  "degreAlcool": "",
  "prixEstime": 0.0,
  "imageEtiquetteUrl": "",
  "tempsGarde": ""
}}
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        content = resp.choices[0].message.content.strip()
        # On enlève d'éventuelles balises ```json
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        return json.loads(content)
    except Exception as e:
        print("❌ GPT Error:", e)
        raise HTTPException(status_code=500, detail="Erreur GPT")


@app.post("/upload-etiquette", response_model=FicheVinResponse)
async def upload_etiquette(file: UploadFile = File(...)):
    try:
        # Lecture de l'image
        data = await file.read()
        image = Image.open(io.BytesIO(data))

        # Pré‑traitement
        image = ameliorer_image(image)

        # OCR Tesseract
        ocr_tess = pytesseract.image_to_string(
            image, lang="fra+eng+ita+spa", config="--oem 1 --psm 6"
        )

        # OCR EasyOCR
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)
        easy_res = reader.readtext(np.array(Image.open(buf)), detail=0)
        ocr_easy = "\n".join(easy_res)

        # Fusionner et nettoyer
        texte_combine = f"Tesseract:\n{ocr_tess}\n\nEasyOCR:\n{ocr_easy}"
        texte_nettoye = nettoyer_texte_ocr(texte_combine)

        # Interroger GPT
        infos = demander_infos_gpt(texte_nettoye)
        infos["ocrTexte"] = ocr_tess

        # S'assurer que prixEstime est un float
        prix = infos.get("prixEstime", 0)
        infos["prixEstime"] = float(prix) if str(prix).replace('.', '', 1).isdigit() else 0.0

        # Forcer tous les champs à str ou valeur par défaut
        for k in ["nom","domaine","appellation","millesime","region","pays","couleur","cepage","degreAlcool","tempsGarde","imageEtiquetteUrl"]:
            if not isinstance(infos.get(k,""), str):
                infos[k] = ""

        return FicheVinResponse(**infos)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
