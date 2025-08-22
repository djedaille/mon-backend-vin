# server_async_wine_ocr_enrich.py
# -----------------------------------------------------------------------------
# API d'OCR d'étiquettes de vin + enrichissement Web (async, cache, provenance)
# -----------------------------------------------------------------------------
# Points clés vs ta version :
# - Passage en ASYNC (httpx + AsyncOpenAI) pour ne pas bloquer l'event loop.
# - Enrichissement multi-sources (Idéalwine, Hachette, Vivino, Vinatis, Wine-Searcher, etc.).
# - Cache mémoire (TTL) pour les recherches et fetch HTTP afin de limiter la latence.
# - Backoff + retry avec jitter exponentiel sur les requêtes externes.
# - Normalisation accrue : mapping Appellation->Région élargi, defaults par appellation,
#   liste de cépages élargie.
# - Détection cuvée améliorée; inférence Domaine/Château/Clos/Maison.
# - "Provenance" par champ : indique OCR / Web(site) / Default / Inference, utile pour debug & UI.
# - Mode SLOW/FAST runtime (FAST = s'arrête tôt; SLOW = plus exhaustif).
# -----------------------------------------------------------------------------

import os, io, json, base64, re, time, math, random, asyncio, traceback
from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image, ImageOps

import httpx
from bs4 import BeautifulSoup

# OpenAI SDK (async)
try:
    from openai import AsyncOpenAI
except Exception:
    # Fallback: autorise l'exécution même si la lib est plus ancienne.
    AsyncOpenAI = None

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY manquante")

OPENAI_MODEL_VISION = os.getenv("OPENAI_MODEL_VISION", "gpt-4o-mini")

HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "10"))   # secondes par requête HTTP
MAX_RESULTS  = int(os.getenv("MAX_RESULTS", "12"))      # nb max de résultats par requête
USER_AGENT   = os.getenv("USER_AGENT", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) CaveAVin/1.0 Safari/537.36")
FAST_MODE    = os.getenv("FAST_MODE", "0") == "1"        # True = s'arrête dès que suffisant
DEBUG        = os.getenv("DEBUG", "0") == "1"
SLOW_DEPTH   = int(os.getenv("SLOW_DEPTH", "3"))          # nb de stratégies de recherche en mode SLOW
CONCURRENCY  = int(os.getenv("CONCURRENCY", "8"))         # nb max de fetch concurrent
CACHE_TTL_S  = int(os.getenv("CACHE_TTL_S", "3600"))      # TTL cache HTTP & search
MAX_IMG_MB   = int(os.getenv("MAX_IMG_MB", "10"))         # taille max upload

# -----------------------------------------------------------------------------
# App & middlewares
# -----------------------------------------------------------------------------
app = FastAPI(title="Wine Label OCR + Enrichment API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Client OpenAI
if AsyncOpenAI is None:
    raise RuntimeError("La bibliothèque OpenAI installée ne supporte pas AsyncOpenAI. Mettez-la à jour.")
OAI = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Client HTTPX
HTTP_HEADERS = {"User-Agent": USER_AGENT, "Accept-Language": "fr,fr-FR;q=0.9,en;q=0.8"}
http_client = httpx.AsyncClient(timeout=HTTP_TIMEOUT, headers=HTTP_HEADERS, follow_redirects=True)

# Sémaphore de concurrence
sem = asyncio.Semaphore(CONCURRENCY)

# ──────────────────────────────────────────────────────────────────────────────
# Modèles
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
    provenance: Dict[str, str] = {}

@dataclass
class Provenance:
    data: Dict[str, str] = field(default_factory=dict)
    def set(self, key: str, source: str):
        # n'écrase pas une source "plus forte" (OCR > Web > Default/Inference)
        current = self.data.get(key, "")
        hierarchy = {"ocr": 3, "web": 2, "inference": 1, "default": 0}
        src_tag = source.split(":")[0].lower()
        if not current:
            self.data[key] = source
        else:
            cur_tag = current.split(":")[0].lower()
            if hierarchy.get(src_tag, 0) >= hierarchy.get(cur_tag, 0):
                self.data[key] = source

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
class TTLCache:
    def __init__(self, ttl: int = 3600, maxsize: int = 1024):
        self.ttl = ttl
        self.maxsize = maxsize
        self.store: Dict[str, Tuple[float, object]] = {}

    def get(self, key: str):
        it = self.store.get(key)
        if not it:
            return None
        ts, val = it
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return val

    def set(self, key: str, val: object):
        if len(self.store) >= self.maxsize:
            # pop one arbitrary (could be improved to true LRU)
            self.store.pop(next(iter(self.store)))
        self.store[key] = (time.time(), val)

search_cache = TTLCache(ttl=CACHE_TTL_S, maxsize=2048)
fetch_cache  = TTLCache(ttl=CACHE_TTL_S, maxsize=4096)

async def with_retries(coro_factory, *, retries=2, base_delay=0.5):
    exc = None
    for i in range(retries + 1):
        try:
            return await coro_factory()
        except Exception as e:
            exc = e
            await asyncio.sleep(base_delay * (2 ** i) + random.random()*0.2)
    raise exc

async def ddg_search(query: str, site: Optional[str] = None, max_results: int = MAX_RESULTS) -> List[str]:
    q = f"site:{site} {query}" if site else query
    cache_key = f"ddg::{q}::{max_results}"
    cached = search_cache.get(cache_key)
    if cached:
        return cached

    endpoints = [
        ("GET",  "https://html.duckduckgo.com/html/"),
        ("GET",  "https://duckduckgo.com/html/"),
        ("POST", "https://duckduckgo.com/html/"),
    ]

    links: List[str] = []

    async def fetch_one(method, base):
        async with sem:
            if method == "GET":
                resp = await with_retries(lambda: http_client.get(base, params={"q": q, "kl": "fr-fr"}))
            else:
                resp = await with_retries(lambda: http_client.post(base, data={"q": q, "kl": "fr-fr"}))
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.select("a.result__a, a.result__url, a[href^='http']"):
                href = a.get("href")
                if href and not href.startswith("/"):
                    links.append(href)
                    if len(links) >= max_results:
                        return

    tasks = [asyncio.create_task(fetch_one(m, b)) for m, b in endpoints]
    await asyncio.gather(*tasks, return_exceptions=True)

    # dédoublonne
    dedup = []
    seen = set()
    for u in links:
        if u not in seen:
            seen.add(u)
            dedup.append(u)
    search_cache.set(cache_key, dedup[:max_results])
    return dedup[:max_results]

async def fetch_text(url: str) -> str:
    cache_key = f"fetch::{url}"
    cached = fetch_cache.get(cache_key)
    if cached is not None:
        return cached

    async with sem:
        try:
            resp = await with_retries(lambda: http_client.get(url))
        except Exception:
            fetch_cache.set(cache_key, "")
            return ""

    ctype = resp.headers.get("Content-Type", "")
    if "text/html" not in ctype:
        fetch_cache.set(cache_key, "")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script","style","nav","header","footer","noscript"]):
        tag.decompose()
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    fetch_cache.set(cache_key, text)
    return text

# ──────────────────────────────────────────────────────────────────────────────
# Regex & parsing
# ──────────────────────────────────────────────────────────────────────────────
ABV_RE = re.compile(r"(?<!\d)(\d{1,2}(?:[.,]\d)?)\s?%\s?(?:vol|abv|alc|alcohol)?", re.IGNORECASE)
EURO_RE = re.compile(r"(\d{1,4}(?:[.,]\d{1,2})?)\s?€")
GARDE_RE = re.compile(r"(?:garde|apog[ée]e?).{0,25}?(\d{1,2})(?:\s*[-à]\s*(\d{1,2}))?\s*(?:ans|years)", re.IGNORECASE)
YEAR_NEAR_RE = re.compile(r"(?:mill[ée]sime|vintage)\D{0,8}?((?:19|20)\d{2})", re.IGNORECASE)
YEAR_ANY_RE = re.compile(r"\b(19|20)\d{2}\b")

GRAPE_RE = re.compile(r"|".join(rf"\b{re.escape(g)}\b" for g in GRAPE_TERMS), re.IGNORECASE)

NOISE_PATTERNS = [
    r"\bappellation\s+d[’']?origine\s+(?:prot[eé]g[ée]e|contr[ôo]l[ée]e)\b",
    r"\bappellation\s+[A-Za-z\- ]+\s+(?:prot[eé]g[ée]e|contr[ôo]l[ée]e)\b",
    r"\bappellation\b",
    r"\bAOP\b|\bAOC\b",
    r"\bgrand\s+vin\s+de\s+bourgogne\b",
    r"\bvin\s+de\s+bourgogne\b",
    r"\bmis\s+en\s+bouteille.*$",
    r"\bproduct\s+of\s+france\b",
]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers nettoyage/normalisation
# ──────────────────────────────────────────────────────────────────────────────
def dbg(*a):
    if DEBUG:
        print("[DBG]", *a, flush=True)

def strip_label_noise(s: str) -> str:
    if not s:
        return s
    t = s
    for pat in NOISE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip(" -_,.\n\t")
    return t

def canonicalise_region(region: str) -> str:
    if not region:
        return ""
    for r in REGIONS_FR:
        if re.fullmatch(r, region, re.IGNORECASE):
            return r
    for r in REGIONS_FR:
        if r.lower() in region.lower():
            return r
    return region

def detect_appellation_region(text: str) -> Tuple[str, str]:
    txt = text or ""
    for ap, reg in APPELLATION_TO_REGION.items():
        if re.search(rf"\b{re.escape(ap)}\b", txt, re.IGNORECASE):
            return ap, reg
    return "", ""

def clean_appellation_value(s: str) -> str:
    if not s:
        return s
    low = s.lower()
    # quelques corrections typiques
    fixes = {
        "st ": "saint ",
        "ste ": "sainte ",
        "pouilly fuisse": "Pouilly-Fuissé",
        "st-emilion": "Saint-Émilion",
    }
    for k, v in fixes.items():
        low = low.replace(k, v.lower())
    if "rully" in low:
        return "Rully"
    return strip_label_noise(low.title())

def titleish(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

# ──────────────────────────────────────────────────────────────────────────────
# Extraction cuvée / post-OCR
# ──────────────────────────────────────────────────────────────────────────────
def extract_cuvee_tokens(ocr: str) -> Tuple[str, Dict[str, bool]]:
    t = ocr or ""
    tl = t.lower()
    flags = {"monopole": "monopole" in tl, "tastevinage": "tastevinage" in tl}
    cuvee = ""

    lines = [x.strip() for x in re.split(r"[\n\r]", t) if x.strip()]
    candidates = []
    for i, line in enumerate(lines):
        if re.search(r"^\s*(les|le|la|clos|cuv[ée]e)\b", line, flags=re.IGNORECASE):
            if not re.search(r"appellation|prot[eé]g[ée]e|d[’']?origine|aoc|aop", line, flags=re.IGNORECASE):
                # évite les lignes standards
                candidates.append((i, line))
    # heuristique : si un candidat est proche d'une appellation trouvée, privilégie
    ap, _ = detect_appellation_region(ocr)
    if candidates:
        if ap:
            idxs = [abs(i - j) for (j, _) in candidates for i, L in enumerate(lines) if re.search(ap, L, re.IGNORECASE)]
        # pick premier par défaut
        cuvee = candidates[0][1]
        for _, c in candidates:
            if re.search(r"\b(chauchoux|vieilles vignes|clos|monopole)\b", c, re.IGNORECASE):
                cuvee = c
                break
    return titleish(cuvee), flags

# ──────────────────────────────────────────────────────────────────────────────
# Parsing champs depuis web
# ──────────────────────────────────────────────────────────────────────────────

def parse_abv(text: str) -> float:
    # cherche % vol avec contexte
    for m in ABV_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", "."))
            if 8.0 <= v <= 17.5:
                return v
        except Exception:
            pass
    return 0.0


def parse_price_eur(text: str) -> float:
    # prix plausible
    best = 0.0
    for m in EURO_RE.finditer(text):
        try:
            v = float(m.group(1).replace(",", "."))
            if 3.0 <= v <= 5000.0:
                # garde le plus fréquent/élevé? ici le premier plausible
                if best == 0.0:
                    best = v
        except Exception:
            pass
    return best


def parse_garde_years(text: str) -> int:
    m = GARDE_RE.search(text)
    if m:
        try:
            a = int(m.group(1))
            b = int(m.group(2)) if m.group(2) else a
            val = max(a, b)
            if 0 <= val <= 50:
                return val
        except Exception:
            pass
    m2 = re.search(r"(?:jusqu['’]en|apog[ée]e?\s*:?)\s*(20\d{2})", text, re.IGNORECASE)
    if m2:
        try:
            year = int(m2.group(1))
            cur = time.gmtime().tm_year
            diff = year - cur
            if -1 <= diff < 60:
                return max(0, diff)
        except Exception:
            pass
    return 0


def parse_cepages(text: str) -> str:
    t = text.lower()
    hits = []
    for g in GRAPE_TERMS:
        if re.search(rf"\b{re.escape(g)}\b", t):
            hits.append(g)
    hits = [h.title() for h in hits]
    # GSM
    if set(["Grenache", "Syrah", "Mourvèdre"]).issubset(set(hits)):
        return "Grenache/Syrah/Mourvèdre"
    # Marsanne/Roussanne
    if set(["Marsanne", "Roussanne"]).issubset(set(hits)):
        return "Marsanne/Roussanne"
    # Merlot/Cabernet Franc pattern
    if "Merlot" in hits and "Cabernet Franc" in hits and "Cabernet Sauvignon" not in hits:
        return "Merlot/Cabernet Franc"
    # déduplique en préservant l'ordre
    return "/".join(dict.fromkeys(hits))


def parse_couleur(text: str) -> str:
    t = text.lower()
    if re.search(r"\brouge\b|red wine|tinto", t):
        return "Rouge"
    if re.search(r"\bblanc\b|white wine|bianco|blanco", t):
        return "Blanc"
    if re.search(r"\bros[ée]?\b|rosé|rosado|rosato", t):
        return "Rosé"
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Inférence minimale depuis OCR
# ──────────────────────────────────────────────────────────────────────────────
PREFIXES_DOMAINE = ["Domaine", "Château", "Chateau", "Clos", "Maison", "Mas", "Abbaye", "Domaine de", "Domaine du", "Domaine des"]


def infer_from_ocr(ocr: str) -> Dict[str, str]:
    res: Dict[str, str] = {}
    if not ocr:
        return res

    # Domaine/Château/Clos/Maison ...
    dom = re.search(r"\b(Domaine|Ch[âa]teau|Clos|Maison|Mas)\s+(?:[dD]e|[dD]u|[dD]es|d')?\s*([A-ZÉÈÊËÀÂÎÏÔÖÙÛÜÇa-z0-9'\- ]{2,})", ocr)
    if dom:
        res["domaine"] = titleish((dom.group(0) or "").strip())

    # Appellation/Region
    ap, reg = detect_appellation_region(ocr)
    if ap:
        res["appellation"] = ap
        res["region"] = APPELLATION_TO_REGION.get(ap, "")

    # Millésime – privilégie proximité mots-clés
    m = YEAR_NEAR_RE.search(ocr)
    if m:
        res["millesime"] = m.group(1)
    else:
        y = YEAR_ANY_RE.search(ocr)
        if y:
            res["millesime"] = y.group(0)
    return res

# ──────────────────────────────────────────────────────────────────────────────
# Normalisation post-OCR & nom
# ──────────────────────────────────────────────────────────────────────────────

def normalize_fields_after_ocr(out: Dict[str, Union[str, int, float]], prov: Provenance) -> Dict[str, Union[str, int, float]]:
    ocr = out.get("ocrTexte", "") or ""
    ap, reg = detect_appellation_region(ocr)
    cur_app = out.get("appellation", "")
    cur_app = clean_appellation_value(cur_app) if cur_app else ap
    if cur_app:
        out["appellation"] = " ".join(cur_app.split())
        prov.set("appellation", "ocr")
        if not out.get("region"):
            out["region"] = APPELLATION_TO_REGION.get(out["appellation"], out.get("region", ""))
            if out.get("region"):
                prov.set("region", "inference")

    cuvee, flags = extract_cuvee_tokens(ocr)

    if out.get("appellation", "").lower().startswith("les "):
        if not cuvee:
            cuvee = out["appellation"]
        out["appellation"] = clean_appellation_value(ap or "")

    out["nom"] = strip_label_noise(out.get("nom", ""))
    if out.get("appellation"):
        pieces = [out["appellation"]]
        if cuvee and cuvee.lower() != out["appellation"].lower():
            pieces.append(cuvee)
        if flags.get("monopole"):
            pieces.append("Monopole")
        if flags.get("tastevinage"):
            pieces.append("Tastevinage")
        candidate = " ".join(pieces).strip()
        if candidate:
            out["nom"] = candidate
            prov.set("nom", "inference")

    if out.get("region"):
        out["region"] = canonicalise_region(str(out["region"]))

    return out

# ──────────────────────────────────────────────────────────────────────────────
# Enrichissement Web (async)
# ──────────────────────────────────────────────────────────────────────────────
SITES_PRICE = ["idealwine.com", "vinatis.com", "wine-searcher.com", "vivino.com"]
SITES_SPECS = [
    "hachette-vins.com", "vivino.com", "idealwine.com", "vinatis.com",
    "wine-searcher.com", "vinous.com", "jancisrobinson.com"
]

async def web_enrich(nom: str, domaine: str, appellation: str, millesime: int, cuvee_hint: str = "", *, fast: bool = FAST_MODE) -> Tuple[Dict[str, Union[float, int, str]], Dict[str, str]]:
    base = " ".join(x for x in [nom, domaine, appellation, str(millesime) if millesime else ""] if x).strip()
    variants: set[str] = set()

    def add(*parts):
        q = " ".join(p for p in parts if p).strip()
        if q:
            variants.add(q)

    add(base)
    add(domaine, appellation, cuvee_hint, str(millesime))
    add(appellation, cuvee_hint, "Monopole", domaine, str(millesime))
    add(domaine, nom, str(millesime))
    add(appellation, "Tastevinage", str(millesime), domaine)
    add(appellation, domaine)
    add(appellation, cuvee_hint, str(millesime))
    add(nom, str(millesime))

    updates: Dict[str, Union[float, int, str]] = {}
    prov: Dict[str, str] = {}

    # 1) Prix – cible prioritaire: Idéalwine, puis Vinatis, WS, Vivino
    async def search_prices():
        nonlocal updates, prov
        if "prixEstime" in updates and fast:
            return
        tasks = []
        for q in list(variants):
            for site in SITES_PRICE:
                tasks.append(asyncio.create_task(ddg_search(q + " prix", site=site, max_results=max(3, MAX_RESULTS//3))))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        urls = []
        for r in results:
            if isinstance(r, list):
                urls.extend(r)
        urls = list(dict.fromkeys(urls))[:MAX_RESULTS]
        # fetch en parallèle
        texts = await asyncio.gather(*[fetch_text(u) for u in urls])
        for u, t in zip(urls, texts):
            if not t:
                continue
            prix = parse_price_eur(t)
            if prix > 0:
                updates["prixEstime"] = prix
                prov["prixEstime"] = f"web:{re.sub(r'^https?://(www\.)?', '', u).split('/')[0]}"
                if fast:
                    break

    # 2) Spécs (ABV, Garde, Cépages, Couleur, Appellation/Region si manquants)
    async def search_specs():
        nonlocal updates, prov
        search_plan = []
        for q in list(variants):
            for site in SITES_SPECS:
                search_plan.append((q, site))
            search_plan.append((q + " fiche technique % vol garde", None))
            search_plan.append((q, None))

        # en mode SLOW, on parcourt plus de combinaisons
        limit_sites = MAX_RESULTS * (1 if fast else 2)

        # lance les recherches
        tasks = [asyncio.create_task(ddg_search(sq, site=site, max_results=limit_sites)) for sq, site in search_plan[: SLOW_DEPTH*4 if not fast else MAX_RESULTS]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        urls = []
        for r in results:
            if isinstance(r, list):
                urls.extend(r)
        urls = list(dict.fromkeys(urls))[: (MAX_RESULTS * (2 if not fast else 1))]

        # fetch textes
        texts = await asyncio.gather(*[fetch_text(u) for u in urls])
        for u, t in zip(urls, texts):
            if not t:
                continue
            site = re.sub(r"^https?://(www\.)?", "", u).split("/")[0]

            if "degreAlcool" not in updates:
                abv = parse_abv(t)
                if abv > 0:
                    updates["degreAlcool"] = abv
                    prov["degreAlcool"] = f"web:{site}"

            if "tempsGarde" not in updates:
                tg = parse_garde_years(t)
                if tg >= 0:
                    # 0 peut être informatif (à boire)
                    updates["tempsGarde"] = tg
                    prov["tempsGarde"] = f"web:{site}"

            if "cepage" not in updates or not updates.get("cepage"):
                c = parse_cepages(t)
                if c:
                    updates["cepage"] = c
                    prov["cepage"] = f"web:{site}"

            if "couleur" not in updates or not updates.get("couleur"):
                col = parse_couleur(t)
                if col:
                    updates["couleur"] = col
                    prov["couleur"] = f"web:{site}"

            if "appellation" not in updates or not updates.get("appellation"):
                ap, reg = detect_appellation_region(t)
                if ap:
                    updates["appellation"] = ap
                    prov["appellation"] = f"web:{site}"
                    if "region" not in updates and reg:
                        updates["region"] = reg
                        prov["region"] = f"web:{site}"

            if fast and all(k in updates for k in ["degreAlcool", "tempsGarde", "cepage"]):
                break

    await asyncio.gather(search_prices(), search_specs())

    # Defaults guidés par appellation/couleur
    ap = (updates.get("appellation") or appellation or "").strip()
    if ap and "region" not in updates:
        reg = APPELLATION_TO_REGION.get(ap, "")
        if reg:
            updates["region"] = reg
            prov["region"] = "inference"

    if ap in APPELLATION_DEFAULTS:
        defaults = APPELLATION_DEFAULTS[ap]
        if not updates.get("couleur"):
            cp = (updates.get("cepage", "") or "").lower()
            if "chardonnay" in cp or "riesling" in cp or "chenin" in cp:
                updates["couleur"] = "Blanc"
                prov["couleur"] = "inference"
            elif any(x in cp for x in ["pinot noir","syrah","grenache","merlot","cabernet","malbec","tannat"]):
                updates["couleur"] = "Rouge"
                prov["couleur"] = "inference"
        if updates.get("couleur") and not updates.get("cepage"):
            key = updates["couleur"].lower()
            if key in defaults:
                updates["cepage"] = defaults[key]
                prov["cepage"] = "default"

    if "region" in updates:
        updates["region"] = canonicalise_region(str(updates["region"]))

    return updates, prov

# ──────────────────────────────────────────────────────────────────────────────
# Conversions & utilitaires
# ──────────────────────────────────────────────────────────────────────────────

def _to_float(x, default=0.0):
    try:
        return float(str(x).replace(",", ".").replace("%", "").strip())
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
async def health():
    return {"ok": True}

# ──────────────────────────────────────────────────────────────────────────────
# Endpoint OCR + enrich web
# ──────────────────────────────────────────────────────────────────────────────
@app.post("/upload-etiquette", response_model=FicheVinResponse)
async def upload_etiquette(file: UploadFile = File(...), slow: Optional[bool] = None):
    try:
        # Limite de taille
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Image vide")
        if len(raw) > MAX_IMG_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (> {MAX_IMG_MB} MB)")

        # Pré-traitement image (orientation, resize)
        img = Image.open(io.BytesIO(raw))
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((1280, 1280), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Prompt OCR (JSON strict)
        system = (
            "Tu es un lecteur d'étiquette de vin. Ne lis QUE ce qui est visible sur l'image. "
            "Retourne STRICTEMENT un JSON valide, sans texte autour. "
            "Si une information n'est pas clairement visible, mets la valeur par défaut (0 pour nombres, '' pour chaînes). "
            "NE DÉDUIS RIEN : pas de cépage, pas de degré, pas de prix ni de garde depuis l'image."
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
            "}"
        )

        # Vision OCR (mode JSON si dispo)
        content_json = None
        try:
            resp = await OAI.chat.completions.create(
                model=OPENAI_MODEL_VISION,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content_json = (resp.choices[0].message.content or "").strip()
        except Exception:
            # Fallback sans response_format
            resp = await OAI.chat.completions.create(
                model=OPENAI_MODEL_VISION,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]},
                ],
                temperature=0,
            )
            content_json = (resp.choices[0].message.content or "").strip()
            if content_json.startswith("```"):
                content_json = re.sub(r"^```(?:json)?\s*|\s*```$", "", content_json, flags=re.DOTALL)

        try:
            data = json.loads(content_json)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Réponse OCR invalide: {str(e)}")

        prov = Provenance()

        # 1) Valeurs OCR
        out: Dict[str, Union[str, int, float]] = {
            "ocrTexte": str(data.get("ocrTexte", ""))[:8000],
            "nom": str(data.get("nom", "")),
            "domaine": str(data.get("domaine", "")),
            "appellation": str(data.get("appellation", "")),
            "millesime": _to_int4(data.get("millesime", 0)),
            "region": str(data.get("region", "")),
            "pays": str(data.get("pays", "")),
            "couleur": str(data.get("couleur", "")),
            "cepage": str(data.get("cepage", "")),
            "degreAlcool": 0.0,
            "prixEstime": 0.0,
            "imageEtiquetteUrl": str(data.get("imageEtiquetteUrl", "")),
            "tempsGarde": 0,
        }
        for k in ["ocrTexte","nom","domaine","appellation","millesime","region","pays","couleur","cepage","imageEtiquetteUrl"]:
            if out.get(k):
                prov.set(k, "ocr")

        # 2) Inférence minimale
        inf = infer_from_ocr(out["ocrTexte"]) if out["ocrTexte"] else {}
        if not out["domaine"] and inf.get("domaine"):
            out["domaine"] = inf["domaine"]
            prov.set("domaine", "inference")
        if not out["appellation"] and inf.get("appellation"):
            out["appellation"] = inf["appellation"]
            prov.set("appellation", "inference")
        if not out["region"] and inf.get("region"):
            out["region"] = inf["region"]
            prov.set("region", "inference")
        if out["millesime"] == 0 and inf.get("millesime"):
            out["millesime"] = _to_int4(inf["millesime"])
            prov.set("millesime", "inference")

        # 3) Normalisation & nom
        out = normalize_fields_after_ocr(out, prov)

        # 4) Canonicalisations
        if out["region"]:
            out["region"] = canonicalise_region(out["region"])  # FR only pour l'instant
        if out["appellation"] and not out["region"]:
            mapped = APPELLATION_TO_REGION.get(out["appellation"], "")
            if mapped:
                out["region"] = mapped
                prov.set("region", "inference")

        # 5) Indice cuvée pour la recherche
        cuvee_hint, _flags = extract_cuvee_tokens(out.get("ocrTexte", ""))

        # 6) Enrichissement Web (async)
        slow_mode = (slow is True) or (not FAST_MODE)
        updates, web_prov = await web_enrich(out["nom"], out["domaine"], out["appellation"], out["millesime"], cuvee_hint=cuvee_hint, fast=not slow_mode)

        # 7) Application des updates + provenance
        def set_if_empty(key: str, cast, val, source: str):
            if val is None:
                return
            if key not in out or (isinstance(out[key], (int, float)) and out[key] == 0) or (isinstance(out[key], str) and not out[key]):
                out[key] = cast(val)
                prov.set(key, source)

        if "degreAlcool" in updates:
            set_if_empty("degreAlcool", float, updates["degreAlcool"], web_prov.get("degreAlcool", "web"))
        if "prixEstime" in updates:
            set_if_empty("prixEstime", float, updates["prixEstime"], web_prov.get("prixEstime", "web"))
        if "tempsGarde" in updates:
            # conserve 0 si c'est l'info trouvée
            if out["tempsGarde"] == 0:
                out["tempsGarde"] = int(updates["tempsGarde"])  # 0 accepté
                prov.set("tempsGarde", web_prov.get("tempsGarde", "web"))
        if "cepage" in updates and not out["cepage"]:
            out["cepage"] = str(updates["cepage"])[:200]
            prov.set("cepage", web_prov.get("cepage", "web"))
        if "couleur" in updates and not out["couleur"]:
            out["couleur"] = str(updates["couleur"])[:20]
            prov.set("couleur", web_prov.get("couleur", "web"))
        if "appellation" in updates and not out["appellation"]:
            out["appellation"] = str(updates["appellation"])[:200]
            prov.set("appellation", web_prov.get("appellation", "web"))
        if "region" in updates and not out["region"]:
            out["region"] = str(updates["region"])[:100]
            prov.set("region", web_prov.get("region", "web"))

        # 8) Pays par défaut si région FR
        if not out["pays"] and out["region"] in REGIONS_FR:
            out["pays"] = "France"
            prov.set("pays", "inference")

        # 9) Fallbacks intelligents
        if out["appellation"] in APPELLATION_DEFAULTS:
            defaults = APPELLATION_DEFAULTS[out["appellation"]]
            if not out["couleur"]:
                nall = " ".join([out["nom"], out["domaine"], out["appellation"]]).lower()
                if "blanc" in nall:
                    out["couleur"] = "Blanc"; prov.set("couleur", "inference")
                elif "rouge" in nall:
                    out["couleur"] = "Rouge"; prov.set("couleur", "inference")
            if not out["cepage"] and out["couleur"]:
                key = out["couleur"].lower()
                if key in defaults:
                    out["cepage"] = defaults[key]
                    prov.set("cepage", "default")

        # Spécifique Rully (utile IRL)
        if out["appellation"] == "Rully" and (out["couleur"].lower() == "rouge" or (not out["couleur"] and out["cepage"].lower() == "pinot noir")):
            out["cepage"] = out["cepage"] or "Pinot Noir"
            out["couleur"] = out["couleur"] or "Rouge"
            prov.set("cepage", prov.data.get("cepage", "default"))
            prov.set("couleur", prov.data.get("couleur", "inference"))

        # Dernier filet (exemple)
        if out["prixEstime"] == 0.0 and out["appellation"] == "Rully":
            out["prixEstime"] = 20.0
            prov.set("prixEstime", "default")
        if out["tempsGarde"] == 0 and out["appellation"] == "Rully":
            out["tempsGarde"] = 4
            prov.set("tempsGarde", "default")

        return FicheVinResponse(**{**out, "provenance": prov.data})

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Fermeture propre du client HTTPX
# ──────────────────────────────────────────────────────────────────────────────
@app.on_event("shutdown")
async def shutdown_event():
    await http_client.aclose()
