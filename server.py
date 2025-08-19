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

        from PIL import Image
        import io, base64, json, traceback

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        system = (
            "Tu lis une étiquette de vin uniquement à partir de l'image."
            " Retourne STRICTEMENT un JSON valide et rien d'autre."
            " Règles simples :"
            " - Si tu vois 'Domaine X', mets domaine='Domaine X'."
            " - Si 'Cuvée Y' apparaît, mets nom='Cuvée Y'."
            " - millesime = 4 chiffres les plus probables (ex: 2018)."
            " - degreAlcool = valeur avec '%' si visible (ex: '13%')."
            " - couleur = rouge / blanc / rosé si déductible (sinon '')."
            " - pays/region/appellation uniquement si visibles ou clairement déductibles."
            " - prixEstime et tempsGarde laissent 0.0 et '' s’ils ne sont pas sur l’étiquette."
            " - N’invente pas : si inconnu => chaîne vide."
            "Pour remplir les champs vide fait une recherche en allant sur le site du domaine ou sur vivino ou autre"
            "Par exemple le tempsGarde tu peux le trouver en cherchant : temps de garde pour vin domaine X, cuvé Y"
            "Le prixEstime tu peux mle trouver sur idealwine"
            "Extrait sans recherches : nom, domaine, appellation, millesime, region, pays, couleur" 
            "Fais ensuite une vraie recherche web pour trouver : cepage, degreAlcool, prixEstime, tempsGarde"
        )

        user_text = (
            "Extrait et renvoie UNIQUEMENT ce JSON :\n"
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
        )

        resp = client.chat.completions.create(
            model="gpt-4o",  # <- au lieu de gpt-4o-mini
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
            temperature=0  # plus déterministe
        )

        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").replace("json\n", "").replace("json", "")
        data = json.loads(content)

        def _safe_float(x, default=0.0):
            try:
                return float(str(x).replace(",", "."))
            except Exception:
                return default

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
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
