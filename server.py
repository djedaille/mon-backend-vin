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
    raise RuntimeError("OPENAI_API_KEY manquante (Render > Settings > Environment)")

client = OpenAI(api_key=OPENAI_API_KEY)
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

@app.get("/health")
def health():
    return {"ok": True}

def _safe_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

@app.post("/upload-etiquette", response_model=FicheVinResponse)
async def upload_etiquette(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw or len(raw) == 0:
            raise HTTPException(status_code=400, detail="Image vide")

        # Convertit en JPEG raisonnable (poids + OCR)
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=400, detail="Fichier non image")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        jpg_bytes = buf.getvalue()
        b64 = base64.b64encode(jpg_bytes).decode("utf-8")

        # Demande à OpenAI : OCR + extraction propre — pas de web-browsing, si inconnu: ""
        system = (
            "Tu lis une étiquette de vin. "
            "Réponds UNIQUEMENT un JSON valide. "
            "Si une info est inconnue, renvoie une chaîne vide. "
            "N'invente pas."
        )
        user_text = (
            "Extrait ces champs et renvoie UNIQUEMENT ce JSON :\n"
            "{\n"
            '  "ocrTexte": "",\n'
            '  "nom": "",\n'
            '  "domaine": "",\n'
            '  "appellation": "",\n'
            '  "millesime": "",\n'
            '  "region": "",\n'
            '  "pays": "",\n'
            '  "couleur": "",\n'
            '  "cepage": "",\n'
            '  "degreAlcool": "",\n'
            '  "prixEstime": 0.0,\n'
            '  "imageEtiquetteUrl": "",\n'
            '  "tempsGarde": ""\n'
            "}\n"
            "Note : millesime et degreAlcool peuvent être des chaînes (ex: '2018', '13%'). "
            "prixEstime est un nombre si possible."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
        )
        content = resp.choices[0].message.content.strip()

        # Nettoyage éventuel de ```json ... ```
        if content.startswith("```"):
            content = content.strip("`")
            content = content.replace("json\n", "").replace("json", "")
        data = json.loads(content)

        # Normalisation types
        out = {
            "ocrTexte": str(data.get("ocrTexte", ""))[:5000],
            "nom": str(data.get("nom", "")),
            "domaine": str(data.get("domaine", "")),
            "appellation": str(data.get("appellation", "")),
            "millesime": str(data.get("millesime", "")),
            "region": str(data.get("region", "")),
            "pays": str(data.get("pays", "")),
            "couleur": str(data.get("couleur", "")),
            "cepage": str(data.get("cepage", "")),
            "degreAlcool": str(data.get("degreAlcool", "")),
            "prixEstime": _safe_float(data.get("prixEstime", 0.0), 0.0),
            "imageEtiquetteUrl": str(data.get("imageEtiquetteUrl", "")),
            "tempsGarde": str(data.get("tempsGarde", "")),
        }
        return FicheVinResponse(**out)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        # Renvoie une erreur JSON explicite (évite un crash silencieux)
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
