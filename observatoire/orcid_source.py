"""
Source de decouverte de talents marocains via l'API publique ORCID.

Contrairement aux sources RSS/Tavily (texte libre analyse par un LLM), ORCID
fournit des donnees STRUCTUREES et auto-declarees par le chercheur lui-meme
(biographie, mots-cles, historique d'affiliations avec pays) -- on peut donc
determiner le pays de residence/activite actuel directement depuis les
donnees de l'API, sans deviner via un LLM. Plus fiable, et economise du
quota Groq puisqu'aucun appel LLM n'est necessaire pour cette source.

API publique ORCID (pub.orcid.org) : gratuite, sans cle obligatoire. Un
enregistrement gratuit sur https://orcid.org/developer-tools donne des
quotas plus confortables (client OAuth), mais le mode public suffit pour
ce volume de requetes.

Necessite le package 'requests' (pip install requests si absent).
"""
import logging
import time
import unicodedata
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("mre_ai")

ORCID_API_BASE = "https://pub.orcid.org/v3.0"
MOROCCO_ISO = "MA"

# Requetes de recherche ORCID (syntaxe Solr) : chercheurs mentionnant une
# origine/nationalite marocaine dans leur profil (biographie, mots-cles...),
# une requete par "secteur" pour repartir les resultats dans les domaines de
# l'Observatoire (memes valeurs que OBSERVATORY_DOMAINS dans app.py).
DOMAIN_QUERIES: Dict[str, str] = {
    "health": '((text:Moroccan OR text:Morocain OR text:Maroc OR text:Morocco OR text:diaspora) AND (keyword:health OR keyword:medicine OR text:chercheur OR text:medical))',
    "education": '((text:Moroccan OR text:Morocain OR text:Maroc OR text:Morocco OR text:diaspora) AND (keyword:education OR text:universite OR text:professeur OR text:enseignant))',
    "tech": '((text:Moroccan OR text:Morocain OR text:Maroc OR text:Morocco OR text:diaspora) AND (keyword:engineering OR keyword:technology OR keyword:AI OR text:startup OR text:developpeur))',
    "agriculture": '((text:Moroccan OR text:Morocain OR text:Maroc OR text:Morocco OR text:diaspora) AND (keyword:agriculture OR keyword:agronomy OR text:agritech OR text:entrepreneur))',
    "industry": '((text:Moroccan OR text:Morocain OR text:Maroc OR text:Morocco OR text:diaspora) AND (keyword:industry OR keyword:manufacturing OR text:ingenieur))',
    "other": '((text:Moroccan OR text:Morocain OR text:Maroc OR text:Morocco OR text:diaspora) AND (keyword:research OR text:talent OR text:laureat OR text:entrepreneur))',
}

MAX_RESULTS_PER_DOMAIN = 8

# Mapping minimal code pays ISO 3166-1 alpha-2 -> nom francais, pour les
# destinations MRE les plus frequentes. Pays non couverts : on garde le code
# ISO tel quel (mieux que rien, ajustable manuellement depuis /admin).
_ISO_TO_FR_COUNTRY = {
    "FR": "France", "CA": "Canada", "US": "Etats-Unis", "GB": "Royaume-Uni",
    "BE": "Belgique", "DE": "Allemagne", "ES": "Espagne", "NL": "Pays-Bas",
    "CH": "Suisse", "IT": "Italie", "AE": "Emirats Arabes Unis", "QA": "Qatar",
    "SA": "Arabie Saoudite", "SE": "Suede", "NO": "Norvege", "DK": "Danemark",
    "AU": "Australie", "JP": "Japon", "SG": "Singapour", "LU": "Luxembourg",
    "PT": "Portugal", "AT": "Autriche", "IE": "Irlande", "FI": "Finlande",
}


def _country_label(iso_code: Optional[str]) -> Optional[str]:
    if not iso_code:
        return None
    return _ISO_TO_FR_COUNTRY.get(iso_code.upper(), iso_code.upper())


def _headers() -> Dict[str, str]:
    return {"Accept": "application/json"}


def _search_orcid_ids(query: str, rows: int) -> List[str]:
    try:
        response = requests.get(
            f"{ORCID_API_BASE}/search/",
            params={"q": query, "rows": rows},
            headers=_headers(),
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        logger.exception("Erreur lors de la recherche ORCID pour la requete '%s'", query)
        return []

    return [
        item["orcid-identifier"]["path"]
        for item in data.get("result") or []
        if item.get("orcid-identifier", {}).get("path")
    ]


def _fetch_json(path: str) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(f"{ORCID_API_BASE}{path}", headers=_headers(), timeout=15)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.warning("Erreur lors de la recuperation ORCID sur %s", path)
        return None


def _extract_person_info(orcid_id: str) -> Dict[str, Any]:
    person = _fetch_json(f"/{orcid_id}/person") or {}

    name_block = person.get("name") or {}
    given = ((name_block.get("given-names") or {}).get("value") or "").strip()
    family = ((name_block.get("family-name") or {}).get("value") or "").strip()
    full_name = f"{given} {family}".strip()

    biography = ((person.get("biography") or {}).get("content") or "").strip()

    keywords = [
        (kw.get("content") or "").strip()
        for kw in (person.get("keywords") or {}).get("keyword", [])
        if kw.get("content")
    ]

    return {"name": full_name, "biography": biography, "keywords": keywords}


def _extract_affiliations(orcid_id: str, endpoint: str) -> List[Dict[str, Any]]:
    """endpoint: 'employments' ou 'educations'. Retourne une liste
    d'affiliations (organisation/pays/dates), triee la plus recente/en cours
    en premier."""
    data = _fetch_json(f"/{orcid_id}/{endpoint}") or {}
    summary_key = f"{endpoint[:-1]}-summary"  # 'employment-summary' / 'education-summary'
    affiliations = []

    for group in data.get("affiliation-group") or []:
        for summary in group.get("summaries") or []:
            entry = summary.get(summary_key)
            if not entry:
                continue
            org = entry.get("organization") or {}
            address = org.get("address") or {}
            start_year = ((entry.get("start-date") or {}).get("year") or {}).get("value")
            end_year = ((entry.get("end-date") or {}).get("year") or {}).get("value")
            affiliations.append({
                "org_name": org.get("name"),
                "country_iso": address.get("country"),
                "start_year": int(start_year) if start_year else None,
                "ongoing": end_year is None,
            })

    affiliations.sort(key=lambda a: (a["ongoing"], a["start_year"] or 0), reverse=True)
    return affiliations


_MOROCCAN_NAME_HINTS = (
    "ayad", "hassani", "taouaf", "khairat", "mouhir", "sadkaoui", "el harrouni",
    "el hassani", "el", "ben", "benn", "ait", "ouali", "ouma", "lahssini",
    "essaid", "amrani", "belhaj", "benslimane", "ziani", "jamai", "elidrissi",
    "belghazi", "elghazouani", "allal", "bennani", "laamrani",
)


def _normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value or "")
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower().strip()


def _looks_moroccan_name(name: str) -> bool:
    normalized = _normalize_text(name)
    if not normalized:
        return False
    return any(hint in normalized for hint in _MOROCCAN_NAME_HINTS)


def _looks_moroccan_origin(person_info: Dict[str, Any], educations: List[Dict[str, Any]]) -> bool:
    """Preuve d'origine marocaine : etudes faites au Maroc (le plus fiable),
    mention explicite dans la bio/mots-cles, ou un nom apparemment marocain.
    On accepte aussi les profils avec une affiliation etrangere explicite,
    car l'objectif est de collecter les MRE, pas seulement les profils qui
    se declarent explicitement marocains dans leur bio."""
    if any(edu.get("country_iso") == MOROCCO_ISO for edu in educations):
        return True

    haystack = " ".join([person_info.get("biography", ""), " ".join(person_info.get("keywords", []))])
    normalized = _normalize_text(haystack)
    if any(marker in normalized for marker in ("moroccan", "marocain", "maroc")):
        return True

    return _looks_moroccan_name(person_info.get("name", ""))


def _current_foreign_country(employments: List[Dict[str, Any]], educations: List[Dict[str, Any]]) -> Optional[str]:
    """Pays du poste actuel ou de la formation la plus recente, si ce n'est PAS le Maroc."""
    for job in employments:
        if job.get("country_iso") and job["country_iso"] != MOROCCO_ISO:
            return job["country_iso"]
    for edu in educations:
        if edu.get("country_iso") and edu["country_iso"] != MOROCCO_ISO:
            return edu["country_iso"]
    return None


def discover_orcid_talents(existing_urls: set | None = None) -> List[Dict[str, Any]]:
    """
    Cherche des chercheurs/academiques marocains residant/actifs a
    l'etranger via l'API publique ORCID, et retourne directement des profils
    structures -- memes cles que le reste du pipeline (title, description,
    source, url, sector, country, expertise_tags, years_experience,
    image_url) -- SANS appel LLM : ORCID fournit deja les donnees
    structurees necessaires pour determiner le pays de facon fiable.
    """
    existing_urls = existing_urls or set()
    discovered: List[Dict[str, Any]] = []
    seen_orcid_ids: set = set()

    for domain, query in DOMAIN_QUERIES.items():
        orcid_ids = _search_orcid_ids(query, MAX_RESULTS_PER_DOMAIN)
        for orcid_id in orcid_ids:
            if orcid_id in seen_orcid_ids:
                continue
            seen_orcid_ids.add(orcid_id)

            profile_url = f"https://orcid.org/{orcid_id}"
            if profile_url in existing_urls:
                continue

            time.sleep(0.3)  # courtoisie envers l'API publique ORCID

            person_info = _extract_person_info(orcid_id)
            if not person_info.get("name"):
                continue

            educations = _extract_affiliations(orcid_id, "educations")
            employments = _extract_affiliations(orcid_id, "employments")

            country_iso = _current_foreign_country(employments, educations)
            if not country_iso:
                logger.info("ORCID: profil ignore (pas de residence a l'etranger confirmee ni formation etrangere) : '%s'", person_info["name"])
                continue

            if not _looks_moroccan_origin(person_info, educations):
                logger.info("ORCID: profil ignore (origine marocaine non confirmee) : '%s'", person_info["name"])
                continue

            current_job = next((j for j in employments if j.get("country_iso") == country_iso), None)
            org_name = (current_job or {}).get("org_name", "")

            description = person_info.get("biography") or (
                f"Chercheur(se) actuellement affilie(e) a {org_name}." if org_name else ""
            )

            discovered.append({
                "title": person_info["name"],
                "description": description[:500],
                "source": "ORCID",
                "url": profile_url,
                "sector": domain,
                "country": _country_label(country_iso),
                "expertise_tags": ", ".join(person_info.get("keywords", [])[:5]),
                "years_experience": None,
                "image_url": "",
            })

    logger.info(
        "ORCID : %s profils MRE decouverts (sur %s identifiants ORCID examines).",
        len(discovered), len(seen_orcid_ids),
    )
    return discovered