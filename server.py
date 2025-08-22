# server.py
import os, io, json, base64, traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquante")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

class FicheVinResponse(BaseModel):
    ocrTexte: str = ""
    nom: str = ""
    domaine: str = ""
    appellation: str = ""
    millesime: int = 0          # 0 = inconnu
    region: str = ""
    pays: str = ""
    couleur: str = ""
    cepage: str = ""
    degreAlcool: float = 0.0    # % ABV (0.0 = inconnu)
    prixEstime: float = 0.0     # € (0.0 = inconnu)
    imageEtiquetteUrl: str = ""
    tempsGarde: int = 0         # années (0 = inconnu)

@app.get("/health")
def health():
    return {"ok": True}

def _to_float(x, default=0.0):
    try:
        return float(str(x).replace(",", ".").replace("%","").strip())
    except Exception:
        return default

def _to_int4(x):
    try:
        s = str(x).strip()
        if len(s) == 4 and s.isdigit():
            return int(s)
        return int(s)
    except Exception:
        return 0

@app.post("/upload-etiquette", response_model=FicheVinResponse)
async def upload_etiquette(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Image vide")

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)  # limite la plus grande dimension à 1024
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        system = (
            "Tu lis une étiquette de vin uniquement à partir de l'image.\n"
            "Retourne STRICTEMENT un JSON valide, sans texte autour.\n"
            "Champs à remplir si visibles, sinon valeur par défaut (0 pour nombres, '' pour chaînes).\n"
            "Rappels:\n"
            "- millesime = 4 chiffres probables.\n"
            "- degreAlcool = nombre (ex: 13.5) sans signe %.\n"
            "- prixEstime = nombre si indiqué (sinon 0).\n"
            "- tempsGarde = nombre d'années si indiqué (sinon 0).\n"
        )

        user_text = (
            "Renvoie UNIQUEMENT ce JSON:\n"
            "{\n"
            '  "ocrTexte": "",\n'
            '  "nom": "",\n'
            '  "domaine": "",\n'
            '  "appellation": "",\n'
            '  "millesime": 0,\n'
            '  "region": "",\n'
            '  "pays": "",\n'
            '  "couleur": "",\n'
            '  "cepage": "",\n'
            '  "degreAlcool": 0.0,\n'
            '  "prixEstime": 0.0,\n'
            '  "imageEtiquetteUrl": "",\n'
            '  "tempsGarde": 0\n'
            "}\n"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            temperature=0
        )

        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").replace("json\n", "").replace("json", "")
        data = json.loads(content)

        out = {
            "ocrTexte": str(data.get("ocrTexte", ""))[:5000],
            "nom": str(data.get("nom", "")),
            "domaine": str(data.get("domaine", "")),
            "appellation": str(data.get("appellation", "")),
            "millesime": _to_int4(data.get("millesime", 0)),
            "region": str(data.get("region", "")),
            "pays": str(data.get("pays", "")),
            "couleur": str(data.get("couleur", "")),
            "cepage": str(data.get("cepage", "")),
            "degreAlcool": _to_float(data.get("degreAlcool", 0.0), 0.0),
            "prixEstime": _to_float(data.get("prixEstime", 0.0), 0.0),
            "imageEtiquetteUrl": str(data.get("imageEtiquetteUrl", "")),
            "tempsGarde": int(str(data.get("tempsGarde", 0)).strip() or 0),
        }
        return FicheVinResponse(**out)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
