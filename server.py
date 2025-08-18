import os, base64, json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # à définir dans Render
OCR_MODEL_DEFAULT = os.getenv("OCR_MODEL", "gpt-4o-mini")  # "gpt-4o" possible
OCR_DETAIL_DEFAULT = os.getenv("OCR_DETAIL", "low")        # "low" ou "high"

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquante (Render → Environment).")

client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="OCR Vin via OpenAI")

# Autorise les appels depuis ton appli mobile / Unity
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod tu peux restreindre
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt():
    # Demande un JSON strict et stable (faible variance)
    return (
        "Tu es un OCR spécialisé vin. Extrait les champs depuis l’étiquette fournie "
        "et retourne STRICTEMENT un JSON (rien d'autre) avec ces clés en snake_case :\n"
        "{\n"
        '  "nom": string,                // nom cuvée ou vin principal (si incertain, "" )\n'
        '  "domaine": string,            // producteur/domaine (si incertain, "" )\n'
        '  "appellation": string,        // AOC/IGP/DO, etc.\n'
        '  "region": string,             // ex: Bordeaux, Rioja\n'
        '  "pays": string,               // ex: France, Espagne\n'
        '  "millesime": string,          // ex: "2018" (laisser "" si absent)\n'
        '  "couleur": string,            // rouge/blanc/rose/petillant ("" si absent)\n'
        '  "cepages": string,            // ex: "Cabernet Sauvignon;Merlot" (concaténer si multiples)\n'
        '  "degre": string,              // ex: "13%"\n'
        '  "volume": string              // ex: "75cl" ou "750ml"\n'
        "}\n"
        "Ne fais aucune remarque, uniquement le JSON. Si une info est absente, mets une chaîne vide."
    )

def ocr_with_openai(image_bytes: bytes, model: str, detail: str) -> dict:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    messages = [
        {"role": "system", "content": "Tu sors uniquement du JSON valide."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt()},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": detail  # "low" recommandé, "high" pour cas difficiles
                    },
                },
            ],
        },
    ]

    # Chat Completions (stable et simple pour Unity)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=400,
    )

    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except Exception as e:
        # Si jamais le modèle a ajouté du texte, tente un rattrapage simple :
        try:
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1:
                return json.loads(txt[start:end+1])
        except:
            pass
        raise HTTPException(status_code=502, detail=f"Réponse non-JSON d’OpenAI: {txt}")

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"ok": True, "service": "ocr-vin-openai"}

@app.post("/upload-etiquette")
async def upload_etiquette(
    image: UploadFile = File(...),
    model: str = OCR_MODEL_DEFAULT,
    detail: str = OCR_DETAIL_DEFAULT
):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier non-image.")
    data = await image.read()
    try:
        result = ocr_with_openai(data, model=model, detail=detail)
        # Ajoute un fail-safe pour toutes les clés attendues
        keys = ["nom","domaine","appellation","region","pays","millesime","couleur","cepages","degre","volume"]
        for k in keys:
            result.setdefault(k, "")
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
