# server.py
import os, io, json, base64, traceback, re, time
from typing import Dict, List, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquante")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()

HTTP_TIMEOUT = 6          # secondes par requête HTTP
MAX_RESULTS  = 5          # nb max de résultats de recherche parcourus
USER_AGENT   = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) CaveAVin/1.0 Safari/537.36"
FAST_MODE    = True       # True = s'arrête au 1er résultat pertinent

# ──────────────────────────────────────────────────────────────────────────────
# Modèle réponse
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Canonicalisation régions / appellations
# ──────────────────────────────────────────────────────────────────────────────
REGIONS_FR = [
    "Alsace","Beaujolais","Bordeaux","Bourgogne","Champagne","Corse","Jura",
    "Languedoc","Loire","Provence","Rhône","Savoie","Sud-Ouest"
]
# mapping d'appellations fréquentes → région (exhaustivité non requise ; tu peux étendre)
APPELLATION_TO_REGION = {
    # Bourgogne
    "Rully":"Bourgogne","Chablis":"Bourgogne","Gevrey-Chambertin":"Bourgogne","Meursault":"Bourgogne",
    "Puligny-Montrachet":"Bourgogne","Volnay":"Bourgogne","Nuits-Saint-Georges":"Bourgogne",
    "Pommard":"Bourgogne","Mercurey":"Bourgogne","Pouilly-Fuissé":"Bourgogne",
    # Bordeaux
    "Margaux":"Bordeaux","Pauillac":"Bordeaux","Saint-Julien":"Bordeaux","Saint-Estèphe":"Bordeaux",
    "Pomerol":"Bordeaux","Saint-Émilion":"Bordeaux","Graves":"Bordeaux","Pessac-Léognan":"Bordeaux",
    # Loire
    "Sancerre":"Loire","Pouilly-Fumé":"Loire","Vouvray":"Loire",
    # Rhône
    "Côte-Rôtie":"Rhône","Hermitage":"Rhône","Crozes-Hermitage":"Rhône","Châteauneuf-du-Pape":"Rhône",
    "Gigondas":"Rhône","Vacqueyras":"Rhône","Cornas":"Rhône",
    # Champagne
    "Champagne":"Champagne",
    # Alsace
    "Alsace":"Alsace","Riesling":"Alsace","Gewurztraminer":"Alsace","Pinot Gris":"Alsace",
    # Provence
    "Bandol":"Provence","Côtes de Provence":"Provence",
    # Beaujolais
    "Morgon":"Beaujolais","Fleurie":"Beaujolais","Moulin-à-Vent":"Beaujolais",
}
# détecte appellation dans un texte et renvoie (appellation, région)
def detect_appellation_region(text: str) -> Tuple[str,str]:
    txt = text or ""
    for ap, reg in APPELLATION_TO_REGION.items():
        if re.search(rf"\b{re.escape(ap)}\b", txt, re.IGNORECASE):
            return ap, reg
    return "", ""

def canonicalise_region(region: str) -> str:
    if not region: return ""
    for r in REGIONS_FR:
        if re.fullmatch(r, region, re.IGNORECASE):
            return r
    # approximation simple
    for r in REGIONS_FR:
        if r.lower() in region.lower():
            return r
    return region

# ──────────────────────────────────────────────────────────────────────────────
# Utilitaires parsing Web
# ──────────────────────────────────────────────────────────────────────────────
SESS = requests.Session()
SESS.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "fr,fr-FR;q=0.9,en;q=0.8"})

ABV_RE = re.compile(r"(?<!\d)(\d{1,2}(?:[.,]\d)?)\s?%")
EURO_RE = re.compile(r"(\d{1,4}(?:[.,]\d{1,2})?)\s?€")
GARDE_RE = re.compile(r"(?:garde|apog[ée]e?).{0,20}?(\d{1,2})(?:\s*[-à]\s*(\d{1,2}))?\s*(?:ans|years)", re.IGNORECASE)

def ddg_search(query: str, site: str=None, max_results: int=MAX_RESULTS) -> List[str]:
    q = query
    if site:
        q = f"site:{site} {query}"
    url = "https://duckduckgo.com/html/"
    try:
        r = SESS.post(url, data={"q": q}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.select("a.result__a, a.result__url"):
            href = a.get("href")
            if not href:
                continue
            # DuckDuckGo renvoie souvent des redirections /l/?kh=...
            if href.startswith("/l/?kh=") or href.startswith("/?q="):
                continue
            links.append(href)
            if len(links) >= max_results:
                break
        return links
    except Exception:
        return []

def fetch_text(url: str) -> str:
    try:
        r = SESS.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        # Evite les binaires
        if "text/html" not in r.headers.get("Content-Type",""):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # retire scripts/styles/nav
        for tag in soup(["script","style","nav","header","footer","noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        return re.sub(r"\n{3,}", "\n\n", text)
    except Exception:
        return ""

def parse_abv(text: str) -> float:
    # renvoie la 1re valeur plausible entre 8% et 17%
    for m in ABV_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", "."))
            if 8.0 <= v <= 17.5:
                return v
        except Exception:
            pass
    return 0.0

def parse_price_eur_on_idealwine(text: str) -> float:
    # simpliste : 1re valeur en € dans la page iDealwine (souvent TTC / enchères)
    for m in EURO_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", "."))
            if 3.0 <= v <= 20000.0:
                return v
        except Exception:
            pass
    return 0.0

def parse_garde_years(text: str) -> int:
    # si "garde 3-5 ans" → 5 ; "apogée 2027-2032" non géré ici simplement
    m = GARDE_RE.search(text)
    if m:
        try:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            val = max(a,b)
            if 1 <= val <= 50:
                return val
        except Exception:
            pass
    # heuristique secondaire : "à boire jusqu'en 2030" → calcule delta
    m2 = re.search(r"(?:jusqu['’]en|apog[ée]e?\s*:?)\s*(20\d{2})", text, re.IGNORECASE)
    if m2:
        try:
            year = int(m2.group(1))
            cur = time.gmtime().tm_year
            if 0 < (year - cur) < 60:
                return max(1, year - cur)
        except Exception:
            pass
    return 0

def web_enrich(nom: str, domaine: str, appellation: str, millesime: int) -> Dict[str, float|int|str]:
    """
    Fait des recherches web et renvoie dict avec éventuels champs trouvés:
    - degreAlcool (float), prixEstime (float, depuis iDealwine), tempsGarde (int), region (str), appellation (str)
    Ne met que ce qui est trouvé de façon plausible ; sinon s'abstient.
    """
    base_query = " ".join(x for x in [nom, domaine, appellation, str(millesime) if millesime else ""] if x).strip()
    if not base_query:
        return {}

    updates: Dict[str, float|int|str] = {}

    # 1) Priorité iDealwine pour le prix
    q_price = base_query + " prix"
    for url in ddg_search(q_price, site="idealwine.com", max_results=MAX_RESULTS or 3):
        text = fetch_text(url)
        if not text: continue
        prix = parse_price_eur_on_idealwine(text)
        if prix > 0:
            updates["prixEstime"] = prix
            if FAST_MODE: break

    # 2) ABV et temps de garde : large (site producteur, cavistes, fiches techniques, vivino…)
    q_specs = base_query + " fiche technique % vol garde"
    visited = 0
    for url in ddg_search(q_specs, site=None, max_results=MAX_RESULTS):
        visited += 1
        text = fetch_text(url)
        if not text: continue
        # ABV
        if "degreAlcool" not in updates:
            abv = parse_abv(text)
            if abv > 0:
                updates["degreAlcool"] = abv
        # garde
        if "tempsGarde" not in updates:
            tg = parse_garde_years(text)
            if tg > 0:
                updates["tempsGarde"] = tg
        # appellation/region hints dans la page
        if "appellation" not in updates or not updates.get("appellation"):
            ap, reg = detect_appellation_region(text)
            if ap:
                updates["appellation"] = ap
                if "region" not in updates and reg:
                    updates["region"] = reg
        # early stop si on a tout
        if FAST_MODE and ("degreAlcool" in updates) and ("tempsGarde" in updates):
            break
        if visited >= MAX_RESULTS:
            break

    # 3) Canonicalisation finale
    if "region" in updates:
        updates["region"] = canonicalise_region(str(updates["region"]))
    # Si on a une appellation sans région, compléter via le mapping
    if "appellation" in updates and "region" not in updates:
        ap = str(updates["appellation"])
        reg = APPELLATION_TO_REGION.get(ap, "")
        if reg:
            updates["region"] = reg

    return updates

# ──────────────────────────────────────────────────────────────────────────────
# Helpers conversion
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Santé
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True}

# ──────────────────────────────────────────────────────────────────────────────
# Endpoint OCR + enrich web
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/upload-etiquette", response_model=FicheVinResponse)
async def upload_etiquette(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Image vide")

        # ↓↓↓ Pré-traitement image pour poids + vitesse
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=75)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Prompt OCR
        system = (
            "Tu lis une étiquette de vin uniquement à partir de l'image.\n"
            "Retourne STRICTEMENT un JSON valide, sans texte autour.\n"
            "Champs à remplir si visibles, sinon valeur par défaut (0 pour nombres, '' pour chaînes).\n"
            "Rappels:\n"
            "- millesime = 4 chiffres probables.\n"
            "- degreAlcool = nombre (ex: 13.5) sans signe %.\n"
            "- prixEstime = 0 (ne PAS remplir depuis l'image tu devrait aller sur les cotes de vin idealwine.com ou sur vivino par exemple).\n"
            "- tempsGarde = 0 (ne PAS remplir depuis l'image fait une recherche internet en tapant temps de garde et tout ce que tu as sur le vin).\n"
            "- cepage = 0 (fait une recherche internet en tapant cepage et le ocr).\n"
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

        # Initial: OCR uniquement (ABV/prix/garde resteront 0 à ce stade)
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
            "degreAlcool": 0.0,              # verrouillé à 0 pour l'étape OCR
            "prixEstime": 0.0,               # idem
            "imageEtiquetteUrl": str(data.get("imageEtiquetteUrl", "")),
            "tempsGarde": 0,                 # idem
        }

        # Canonicalisation basique via texte OCR
        if not out["appellation"]:
            ap, reg = detect_appellation_region(out["ocrTexte"])
            if ap:
                out["appellation"] = ap
                if not out["region"] and reg:
                    out["region"] = reg
        # Canonicalise la région si déjà renseignée
        if out["region"]:
            out["region"] = canonicalise_region(out["region"])
        # Si on a une appellation avec mapping, complète la région
        if out["appellation"] and not out["region"]:
            mapped = APPELLATION_TO_REGION.get(out["appellation"], "")
            if mapped:
                out["region"] = mapped

        # ── Enrichissement Web (ABV + prix iDealwine + temps de garde + éventuelle better appellation/région)
        updates = web_enrich(out["nom"], out["domaine"], out["appellation"], out["millesime"])

        # On applique : on NE met que ce qui a été vraiment trouvé
        if "degreAlcool" in updates:
            out["degreAlcool"] = float(updates["degreAlcool"])
        if "prixEstime" in updates:
            out["prixEstime"] = float(updates["prixEstime"])
        if "tempsGarde" in updates:
            out["tempsGarde"] = int(updates["tempsGarde"])
        if "appellation" in updates and not out["appellation"]:
            out["appellation"] = str(updates["appellation"])
        if "region" in updates and not out["region"]:
            out["region"] = str(updates["region"])

        # Pays par défaut si région FR
        if not out["pays"] and out["region"] in REGIONS_FR:
            out["pays"] = "France"

        return FicheVinResponse(**out)

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {e}")
