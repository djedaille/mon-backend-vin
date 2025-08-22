# server.py
# -----------------------------------------------------------------------------
# API OCR d'étiquettes de vin + enrichissement web (simple, style ancien script)
# -----------------------------------------------------------------------------
# - /health : test rapide
# - /upload-etiquette : POST multipart avec "file"
#   → OCR via OpenAI Vision (gpt-4o-mini)
#   → Inférence minimale + enrichissement web (prix, cépage, %vol, garde, etc.)
#   → Sortie : FicheVinResponse (simple)
# -----------------------------------------------------------------------------

import os, io, json, base64, re, time, traceback
from typing import Dict, List, Tuple, Union

from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquante")

client = OpenAI(api_key=OPENAI_API_KEY)
OPENAI_MODEL_VISION = "gpt-4o-mini"

HTTP_TIMEOUT = 10
MAX_RESULTS  = 12
USER_AGENT   = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) CaveAVin/1.0 Safari/537.36"
FAST_MODE    = False

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI()

# ──────────────────────────────────────────────────────────────────────────────
# Modèle de sortie
# ──────────────────────────────────────────────────────────────────────────────
class FicheVinResponse(BaseModel):
    ocrTexte: str = ""
    nom: str = ""
    domaine: str = ""
    appellation: str = ""
    millesime: int = 0
    region: str = ""
    pays: str = ""
    couleur: str = ""
    cepage: str = ""
    degreAlcool: float = 0.0
    prixEstime: float = 0.0
    imageEtiquetteUrl: str = ""
    tempsGarde: int = 0
# ──────────────────────────────────────────────────────────────────────────────
# Data – mappings étendus (non exhaustifs mais fournis "à fond")
# ──────────────────────────────────────────────────────────────────────────────
REGIONS_FR = [
    "Alsace","Beaujolais","Bordeaux","Bourgogne","Champagne","Corse","Jura",
    "Languedoc","Loire","Provence","Rhône","Savoie","Sud-Ouest",
]

# Cartographie étendue d'appellations (FR focus)
APPELLATION_TO_REGION: Dict[str, str] = {
    # Bourgogne (Côte d'Or, Côte Chalonnaise, Mâconnais, Chablisien)
    "Chablis": "Bourgogne", "Petit Chablis": "Bourgogne", "Chablis Grand Cru": "Bourgogne",
    "Irancy": "Bourgogne", "Saint-Bris": "Bourgogne",
    "Côte de Nuits": "Bourgogne", "Gevrey-Chambertin": "Bourgogne", "Morey-Saint-Denis": "Bourgogne",
    "Chambolle-Musigny": "Bourgogne", "Vougeot": "Bourgogne", "Vosne-Romanée": "Bourgogne",
    "Nuits-Saint-Georges": "Bourgogne", "Marsannay": "Bourgogne",
    "Aloxe-Corton": "Bourgogne", "Savigny-lès-Beaune": "Bourgogne", "Beaune": "Bourgogne",
    "Pommard": "Bourgogne", "Volnay": "Bourgogne", "Meursault": "Bourgogne",
    "Puligny-Montrachet": "Bourgogne", "Chassagne-Montrachet": "Bourgogne", "Saint-Aubin": "Bourgogne",
    "Santenay": "Bourgogne", "Corton": "Bourgogne", "Corton-Charlemagne": "Bourgogne",
    "Mercurey": "Bourgogne", "Rully": "Bourgogne", "Givry": "Bourgogne", "Montagny": "Bourgogne",
    "Bouzeron": "Bourgogne", "Mâcon": "Bourgogne", "Mâcon-Villages": "Bourgogne",
    "Pouilly-Fuissé": "Bourgogne", "Saint-Véran": "Bourgogne", "Viré-Clessé": "Bourgogne",

    # Beaujolais
    "Beaujolais": "Beaujolais", "Beaujolais-Villages": "Beaujolais",
    "Morgon": "Beaujolais", "Fleurie": "Beaujolais", "Moulin-à-Vent": "Beaujolais",
    "Chiroubles": "Beaujolais", "Juliénas": "Beaujolais", "Chénas": "Beaujolais",
    "Brouilly": "Beaujolais", "Côte de Brouilly": "Beaujolais", "Saint-Amour": "Beaujolais",

    # Bordeaux (rive gauche/droite)
    "Médoc": "Bordeaux", "Haut-Médoc": "Bordeaux", "Pauillac": "Bordeaux", "Saint-Julien": "Bordeaux",
    "Saint-Estèphe": "Bordeaux", "Margaux": "Bordeaux", "Pessac-Léognan": "Bordeaux", "Graves": "Bordeaux",
    "Sauternes": "Bordeaux", "Barsac": "Bordeaux",
    "Saint-Émilion": "Bordeaux", "Saint-Emilion": "Bordeaux", "Pomerol": "Bordeaux",
    "Lalande-de-Pomerol": "Bordeaux", "Fronsac": "Bordeaux", "Canon-Fronsac": "Bordeaux",
    "Blaye": "Bordeaux", "Bourg": "Bordeaux", "Entre-Deux-Mers": "Bordeaux",

    # Loire
    "Sancerre": "Loire", "Pouilly-Fumé": "Loire", "Menetou-Salon": "Loire", "Quincy": "Loire",
    "Reuilly": "Loire", "Coteaux du Giennois": "Loire", "Chinon": "Loire", "Bourgueil": "Loire",
    "Saint-Nicolas-de-Bourgueil": "Loire", "Saumur": "Loire", "Saumur-Champigny": "Loire",
    "Vouvray": "Loire", "Montlouis-sur-Loire": "Loire", "Muscadet": "Loire",
    "Savennières": "Loire", "Anjou": "Loire", "Coteaux du Layon": "Loire",

    # Rhône
    "Côte-Rôtie": "Rhône", "Hermitage": "Rhône", "Crozes-Hermitage": "Rhône", "Cornas": "Rhône",
    "Saint-Joseph": "Rhône", "Condrieu": "Rhône", "Châteauneuf-du-Pape": "Rhône",
    "Gigondas": "Rhône", "Vacqueyras": "Rhône", "Tavel": "Rhône",
    "Côtes-du-Rhône": "Rhône", "Côtes-du-Rhône Villages": "Rhône",

    # Languedoc/Roussillon
    "Picpoul de Pinet": "Languedoc", "Faugères": "Languedoc", "Saint-Chinian": "Languedoc",
    "Minervois": "Languedoc", "Corbières": "Languedoc", "Fitou": "Languedoc",
    "Collioure": "Languedoc", "Banyuls": "Languedoc",

    # Provence
    "Bandol": "Provence", "Côtes de Provence": "Provence", "Coteaux d'Aix-en-Provence": "Provence",
    "Coteaux Varois": "Provence", "Palette": "Provence", "Cassis": "Provence",

    # Champagne
    "Champagne": "Champagne",

    # Alsace
    "Alsace": "Alsace", "Alsace Grand Cru": "Alsace", "Crémant d'Alsace": "Alsace",

    # Jura
    "Arbois": "Jura", "Château-Chalon": "Jura", "Côtes du Jura": "Jura",

    # Savoie/Bugey
    "Vin de Savoie": "Savoie", "Roussette de Savoie": "Savoie", "Bugey": "Savoie",

    # Sud-Ouest
    "Cahors": "Sud-Ouest", "Madiran": "Sud-Ouest", "Gaillac": "Sud-Ouest", "Bergerac": "Sud-Ouest",
    "Pécharmant": "Sud-Ouest", "Jurançon": "Sud-Ouest",

    # Corse
    "Ajaccio": "Corse", "Patrimonio": "Corse", "Figari": "Corse",
}

# Defaults par appellation (indicatifs – élargis)
APPELLATION_DEFAULTS: Dict[str, Dict[str, str]] = {
    # Bourgogne
    "Rully": {"rouge": "Pinot Noir", "blanc": "Chardonnay"},
    "Mercurey": {"rouge": "Pinot Noir", "blanc": "Chardonnay"},
    "Gevrey-Chambertin": {"rouge": "Pinot Noir"},
    "Nuits-Saint-Georges": {"rouge": "Pinot Noir"},
    "Pommard": {"rouge": "Pinot Noir"},
    "Volnay": {"rouge": "Pinot Noir"},
    "Meursault": {"blanc": "Chardonnay"},
    "Chablis": {"blanc": "Chardonnay"},
    "Pouilly-Fuissé": {"blanc": "Chardonnay"},
    "Saint-Véran": {"blanc": "Chardonnay"},

    # Loire
    "Sancerre": {"blanc": "Sauvignon Blanc", "rouge": "Pinot Noir"},
    "Pouilly-Fumé": {"blanc": "Sauvignon Blanc"},
    "Vouvray": {"blanc": "Chenin"},
    "Chinon": {"rouge": "Cabernet Franc"},
    "Bourgueil": {"rouge": "Cabernet Franc"},

    # Bordeaux
    "Pauillac": {"rouge": "Cabernet Sauvignon/Merlot"},
    "Margaux": {"rouge": "Cabernet Sauvignon/Merlot"},
    "Saint-Émilion": {"rouge": "Merlot/Cabernet Franc"},
    "Pomerol": {"rouge": "Merlot/Cabernet Franc"},

    # Rhône
    "Côte-Rôtie": {"rouge": "Syrah"},
    "Hermitage": {"rouge": "Syrah", "blanc": "Marsanne/Roussanne"},
    "Crozes-Hermitage": {"rouge": "Syrah", "blanc": "Marsanne/Roussanne"},
    "Cornas": {"rouge": "Syrah"},
    "Saint-Joseph": {"rouge": "Syrah", "blanc": "Marsanne/Roussanne"},
    "Condrieu": {"blanc": "Viognier"},
    "Châteauneuf-du-Pape": {"rouge": "Grenache/Syrah/Mourvèdre"},
    "Gigondas": {"rouge": "Grenache/Syrah/Mourvèdre"},
    "Vacqueyras": {"rouge": "Grenache/Syrah/Mourvèdre"},

    # Provence
    "Bandol": {"rouge": "Mourvèdre/Grenache/Cinsault", "rosé": "Mourvèdre/Grenache/Cinsault"},
    "Côtes de Provence": {"rosé": "Grenache/Cinsault/Syrah"},

    # Jura
    "Arbois": {"blanc": "Savagnin/Chardonnay", "rouge": "Trousseau/Poulsard/Pinot Noir"},

    # Sud-Ouest
    "Cahors": {"rouge": "Malbec/Merlot"},
    "Madiran": {"rouge": "Tannat/Cabernet Franc/Cabernet Sauvignon"},

    # Alsace
    "Alsace": {"blanc": "Riesling/Gewurztraminer/Pinot Gris/Muscat/Sylvaner/Pinot Blanc"},
}

# Liste élargie de cépages (fr + intl)
GRAPE_TERMS = [
    # FR classiques
    "pinot noir","chardonnay","aligoté","gamay","sauvignon","sauvignon blanc","chenin",
    "syrah","grenache","mourvèdre","carignan","cinsault","merlot","cabernet sauvignon",
    "cabernet franc","viognier","riesling","gewurztraminer","pinot gris","sémillon","ugni blanc",
    "savagnin","trousseau","poulsard","pinot blanc","marsanne","roussanne","muscat","sylvaner",
    # Intl
    "tempranillo","sangiovese","nebbiolo","barbera","aglianico","primitivo","zinfandel",
    "touriga nacional","touriga franca","alvarinho","vermentino","grüner veltliner","malbec",
    "petit verdot","mencia","glera","trebbiano","montepulciano","carignan","mourvedre",
]

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires & cache TTL
# ──────────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────
# Utils OCR / Web
# ──────────────────────────────────────────────────────────────────────────────
SESS = requests.Session()
SESS.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "fr,fr-FR;q=0.9,en;q=0.8"})

ABV_RE = re.compile(r"(?<!\d)(\d{1,2}(?:[.,]\d)?)\s?%\s?(?:vol|abv|alc)?", re.IGNORECASE)
EURO_RE = re.compile(r"(\d{1,4}(?:[.,]\d{1,2})?)\s?€")
GARDE_RE = re.compile(r"(?:garde|apog[ée]e?).{0,25}?(\d{1,2})(?:\s*[-à]\s*(\d{1,2}))?\s*(?:ans|years)", re.IGNORECASE)

def ddg_search(query: str, site: str=None, max_results: int=MAX_RESULTS) -> List[str]:
    q = f"site:{site} {query}" if site else query
    urls = []
    try:
        r = SESS.post("https://duckduckgo.com/html/", data={"q": q, "kl":"fr-fr"}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.select("a.result__a, a[href^='http']"):
            href = a.get("href")
            if href and not href.startswith("/"):
                urls.append(href)
                if len(urls) >= max_results:
                    break
    except Exception:
        pass
    return urls

def fetch_text(url: str) -> str:
    try:
        r = SESS.get(url, timeout=HTTP_TIMEOUT)
        if "text/html" not in r.headers.get("Content-Type",""):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","header","footer","noscript"]):
            tag.decompose()
        return soup.get_text(" ", strip=True)
    except Exception:
        return ""

def parse_abv(text: str) -> float:
    m = ABV_RE.search(text)
    if m:
        v = float(m.group(1).replace(",", "."))
        if 8.0 <= v <= 17.5:
            return v
    return 0.0

def parse_price(text: str) -> float:
    m = EURO_RE.search(text)
    if m:
        v = float(m.group(1).replace(",", "."))
        if 3.0 <= v <= 5000.0:
            return v
    return 0.0

def parse_garde(text: str) -> int:
    m = GARDE_RE.search(text)
    if m:
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        return max(a,b)
    return 0

def parse_cepages(text: str) -> str:
    hits = []
    for g in GRAPE_TERMS:
        if re.search(rf"\b{re.escape(g)}\b", text.lower()):
            hits.append(g.title())
    if set(["Grenache","Syrah","Mourvèdre"]).issubset(hits):
        return "Grenache/Syrah/Mourvèdre"
    return "/".join(dict.fromkeys(hits))

def parse_couleur(text: str) -> str:
    t = text.lower()
    if "rouge" in t or "red wine" in t: return "Rouge"
    if "blanc" in t or "white wine" in t: return "Blanc"
    if "rosé" in t or "rosado" in t: return "Rosé"
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/upload-etiquette", response_model=FicheVinResponse)
async def upload_etiquette(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Image vide")

        # Pré-traitement image
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((1280, 1280), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Prompt OCR
        system = "Tu es un lecteur d'étiquette de vin. Ne lis QUE ce qui est visible sur l'image. Retourne STRICTEMENT un JSON."
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
            "}"
        )

        resp = client.chat.completions.create(
            model=OPENAI_MODEL_VISION,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [
                    {"type":"text","text":user_text},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]}
            ],
            temperature=0
        )

        content = (resp.choices[0].message.content or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.DOTALL)
        data = json.loads(content)

        # OCR
        out = FicheVinResponse(**data)

        # Enrichissement web
        q = " ".join([out.nom,out.domaine,out.appellation,str(out.millesime or "")]).strip()
        urls = []
        for site in ["idealwine.com","vivino.com","vinatis.com","wine-searcher.com"]:
            urls += ddg_search(q, site=site, max_results=MAX_RESULTS)
        texts = [fetch_text(u) for u in urls]

        for t in texts:
            if not out.degreAlcool:
                out.degreAlcool = parse_abv(t)
            if not out.prixEstime:
                out.prixEstime = parse_price(t)
            if not out.tempsGarde:
                out.tempsGarde = parse_garde(t)
            if not out.cepage:
                out.cepage = parse_cepages(t)
            if not out.couleur:
                out.couleur = parse_couleur(t)

        # Defaults
        if out.appellation in APPELLATION_DEFAULTS:
            defs = APPELLATION_DEFAULTS[out.appellation]
            if not out.couleur and defs:
                if "blanc" in defs: out.couleur = "Blanc"
                elif "rouge" in defs: out.couleur = "Rouge"
            if not out.cepage and out.couleur.lower() in defs:
                out.cepage = defs[out.couleur.lower()]

        if out.region in REGIONS_FR and not out.pays:
            out.pays = "France"

        return out

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {e}")
