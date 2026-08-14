"""
Agent LangChain (tool-calling) de decouverte de talents MRE.

C'est le LLM lui-meme qui choisit, a chaque etape, quelle source interroger
(Tavily par secteur, Tavily actualites, RSS) et quand s'arreter, au lieu de
tout parcourir betement comme l'ancienne boucle for.

Economie de quota OpenRouter (pool Observatoire) :
  - la recherche de sources (Tavily, RSS) NE CONSOMME AUCUN token OpenRouter ;
  - la verification de doublon est une comparaison de chaines locale ;
  - le SEUL outil qui consomme un appel OpenRouter "metier" est
    `validate_and_store_profile` (reutilise extract_talent_profile(), deja
    protege par son propre retry multi-cles via openrouter_key_pool.py) ;
  - le raisonnement de l'agent lui-meme (choix d'outil a chaque tour)
    consomme AUSSI un appel OpenRouter par tour -- c'est pourquoi
    RetryableChatOpenRouter est utilise ici : une seule cle qui rate-limite
    ne doit pas faire echouer tout le cycle de sync.

Garde-fous (inchanges par rapport a la version Groq) :
  - RetryableChatOpenRouter (retry multi-cles) au lieu d'une seule cle figee ;
  - compteur CODE (pas juste une consigne texte) qui plafonne le nombre
    de validate_and_store_profile par cycle a MAX_VALIDATIONS ;
  - delai de courtoisie apres chaque validation reussie, pour eviter
    d'enchainer plusieurs appels OpenRouter "lourds" a la suite et risquer un
    429 meme avec plusieurs cles ;
  - MAX_ITERATIONS coherent avec l'objectif annonce (5-8 profils valides) :
    chaque profil coute au minimum ~3 tours (recherche + check_duplicate +
    validate).

Les resultats bruts (title/content/url/source) recuperes par les outils de
recherche ne sont PAS renvoyes en entier au LLM : ils sont mis en cache
localement (par URL) et seul un resume court est montre a l'agent. Quand
l'agent valide un candidat, il ne passe que l'URL ; le texte complet est
relu depuis le cache pour l'appel a l'extracteur LLM.
"""
import json
import logging
import os
import time
from typing import Any, Dict, List, Set

from .langchain_compat import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

from .retryable_openrouter import RetryableChatOpenRouter
from .talent_extractor_llm import extract_talent_profile
from .talent_web_search import DOMAIN_QUERIES, search_talents_web, search_talent_articles
from .talent_monitor import _fetch_rss_raw_entries, _is_likely_duplicate, _normalize_name
from .orcid_source import discover_orcid_talents
from config import get_settings

logger = logging.getLogger("sanad.ai")

# Chaque profil valide coute au minimum 3 tours (recherche + check_duplicate
# + validate), plus les tours "d'exploration" qui ne debouchent sur rien.
MAX_ITERATIONS = 18

# Garde-fou CODE (pas juste une consigne texte dans le prompt) : meme si le
# LLM ignore la consigne "arrete-toi apres 5-8 profils", ce compteur ferme
# l'outil des que la limite est atteinte.
MAX_VALIDATIONS = 8

# Delai de courtoisie apres chaque validation reussie (appel OpenRouter
# "lourd") pour eviter d'enchainer plusieurs appels consecutifs et risquer
# un 429, meme avec plusieurs cles dans le pool.
POST_VALIDATION_SLEEP_SECONDS = 1.0

MAX_SNIPPET_CHARS = 150
MAX_TOOL_RESULTS_SHOWN = 8

SYSTEM_PROMPT = """Tu es un agent de veille qui cherche des Marocains Residant a l'Etranger \
(MRE) talentueux (chercheurs, entrepreneurs, ingenieurs, experts...) pour l'Observatoire des \
Talents Marocains.

Outils disponibles :
- des outils de RECHERCHE (search_sector, search_news, search_rss, search_orcid) : gratuits, \
pas de cout, utilise-les pour explorer.
- check_duplicate : verification locale gratuite, a utiliser AVANT de valider un candidat.
- validate_and_store_profile : SEUL outil qui coute une requete IA. A n'utiliser que sur un \
candidat serieux (nom propre identifiable, indice clair de residence/activite a l'etranger). \
Passe-lui l'URL exacte renvoyee par un outil de recherche. Limite a 8 validations par cycle --
l'outil te previendra si tu atteins la limite.

Consignes :
1. Varie les sources et les secteurs a chaque etape (essaie plusieurs secteurs differents avec \
search_sector avant de conclure qu'il n'y a rien), ne repete pas une recherche deja faite.
2. Verifie toujours check_duplicate avant validate_and_store_profile.
3. Arrete-toi des que tu as valide 5 a 8 nouveaux profils, ou des que plusieurs recherches \
d'affilee n'apportent plus rien de nouveau.
4. Sois direct et econome : pas de blabla, enchaine les appels d'outils utiles."""


def _entry_summary(entry: Dict[str, Any]) -> str:
    snippet = (entry.get("content") or "")[:MAX_SNIPPET_CHARS].strip()
    return f"- titre: {entry.get('title', '')}\n  url: {entry.get('url', '')}\n  extrait: {snippet}"


def _summarize_entries(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "Aucun resultat."
    shown = entries[:MAX_TOOL_RESULTS_SHOWN]
    lines = [_entry_summary(e) for e in shown]
    remainder = len(entries) - len(shown)
    if remainder > 0:
        lines.append(f"(+ {remainder} autre(s) resultat(s) non affiches)")
    return "\n".join(lines)


def _build_tools(
    existing_urls: Set[str],
    known_normalized_names: Set[str],
    discovered: List[Dict[str, Any]],
    raw_cache: Dict[str, Dict[str, Any]],
):
    """Cree les outils LangChain de cette execution, avec etat capture par
    closure (cache des entrees brutes, doublons connus, liste des profils
    valides, compteur de validations) -- necessaire car chaque cycle de
    sync doit repartir a zero."""

    validation_count = {"n": 0}

    def _cache_new_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fresh = []
        for entry in entries:
            url = entry.get("url") or ""
            if not url or url in existing_urls or url in raw_cache:
                continue
            raw_cache[url] = entry
            fresh.append(entry)
        return fresh

    @tool
    def search_sector(domain: str) -> str:
        """Recherche web (Tavily) de talents MRE dans UN secteur donne. Domaines
        valides : tech, health, education, agriculture, industry, finance,
        creative, social, other. Ne consomme aucun quota IA. Retourne une
        courte liste titre/url/extrait -- utilise l'url exacte pour valider
        un candidat ensuite."""
        domain = (domain or "").strip().lower()
        if domain not in DOMAIN_QUERIES:
            return f"Domaine inconnu '{domain}'. Domaines valides : {', '.join(DOMAIN_QUERIES.keys())}."
        try:
            entries = search_talents_web(domains=[domain])
        except Exception:
            logger.exception("Erreur outil search_sector pour le domaine '%s'", domain)
            return "Erreur lors de la recherche web pour ce secteur."
        fresh = _cache_new_entries(entries)
        return _summarize_entries(fresh)

    @tool
    def search_news(_dummy: str = "") -> str:
        """Recherche des articles de presse recents (60 derniers jours) parlant
        de talents marocains a l'etranger (distinctions, succes, publications).
        Ne consomme aucun quota IA. Retourne une courte liste titre/url/extrait."""
        try:
            entries = search_talent_articles()
        except Exception:
            logger.exception("Erreur outil search_news")
            return "Erreur lors de la recherche d'articles de presse."
        fresh = _cache_new_entries(entries)
        return _summarize_entries(fresh)

    @tool
    def search_rss(_dummy: str = "") -> str:
        """Parcourt les flux RSS/Atom configures (data/talent_sources.json) a la
        recherche d'articles mentionnant un talent marocain. Ne consomme
        aucun quota IA. Retourne une courte liste titre/url/extrait."""
        try:
            entries = _fetch_rss_raw_entries()
        except Exception:
            logger.exception("Erreur outil search_rss")
            return "Erreur lors de la lecture des flux RSS."
        fresh = _cache_new_entries(entries)
        return _summarize_entries(fresh)

    @tool
    def search_orcid(_dummy: str = "") -> str:
        """Interroge l'API ORCID pour des profils structurés de chercheurs
        marocains actifs ou résidant à l'étranger. Ne consomme aucun quota IA
        et retourne une courte liste titre/url/extrait."""
        try:
            entries = discover_orcid_talents(existing_urls=existing_urls)
        except Exception:
            logger.exception("Erreur outil search_orcid")
            return "Erreur lors de la lecture des profils ORCID."
        fresh = _cache_new_entries(entries)
        return _summarize_entries(fresh)

    @tool
    def check_duplicate(name: str) -> str:
        """Verifie SANS appel IA si un nom ressemble a un profil deja connu
        (deja en base ou deja valide dans cette session). Utilise ceci avant
        validate_and_store_profile pour ne pas gaspiller de quota sur un
        doublon evident."""
        if _is_likely_duplicate(name or "", known_normalized_names):
            return f"Doublon probable pour '{name}' -- ne pas valider."
        return f"Pas de doublon connu pour '{name}'."

    @tool
    def validate_and_store_profile(url: str) -> str:
        """Valide DEFINITIVEMENT un candidat via l'agent d'extraction IA (SEUL
        outil qui consomme un appel OpenRouter) puis l'ajoute a la liste des
        profils decouverts si retenu. `url` doit etre une URL EXACTE renvoyee
        par un outil de recherche precedent (search_sector, search_news,
        search_rss). Ne l'utilise que sur des candidats serieux, apres
        check_duplicate. Limite a 8 validations par cycle."""
        if validation_count["n"] >= MAX_VALIDATIONS:
            return (
                f"Limite de {MAX_VALIDATIONS} validations atteinte pour ce cycle -- "
                "arrete-toi maintenant, n'appelle plus cet outil."
            )

        entry = raw_cache.get((url or "").strip())
        if not entry:
            return "URL inconnue -- utilise une url exacte renvoyee par un outil de recherche."

        try:
            profile = extract_talent_profile(
                title=entry.get("title", ""),
                text=entry.get("content", ""),
                source_name=entry.get("source", "Source externe"),
                source_url=entry.get("url", ""),
            )
        except Exception:
            logger.exception("Erreur lors de la validation IA pour '%s'", entry.get("title"))
            return "Erreur technique lors de la validation -- candidat ignore."

        if not profile:
            return "Candidat rejete par la validation IA (pas un talent MRE confirme)."

        if _is_likely_duplicate(profile["title"], known_normalized_names):
            return f"Rejete : '{profile['title']}' est un doublon probable."

        known_normalized_names.add(_normalize_name(profile["title"]))
        discovered.append({
            "title": profile["title"],
            "description": profile["description"],
            "source": profile["source_name"],
            "url": profile["source_url"],
            "sector": profile["sector"],
            "country": profile["country"],
            "expertise_tags": profile["expertise_tags"],
            "years_experience": profile["years_experience"],
            "image_url": profile.get("image_url", ""),
        })
        validation_count["n"] += 1

        # Delai de courtoisie apres un appel OpenRouter "lourd" -- evite
        # d'enchainer plusieurs validations consecutives et de risquer un 429.
        time.sleep(POST_VALIDATION_SLEEP_SECONDS)

        remaining = MAX_VALIDATIONS - validation_count["n"]
        return (
            f"Profil valide et enregistre : '{profile['title']}' ({profile['country']}). "
            f"({remaining} validation(s) restante(s) pour ce cycle.)"
        )

    return [search_sector, search_news, search_rss, search_orcid, check_duplicate, validate_and_store_profile]


def _build_agent_executor(tools) -> AgentExecutor:
    # RetryableChatOpenRouter gere la rotation de cles en interne (retry sur
    # 401/429 a chaque tour de raisonnement) -- plus besoin de recuperer
    # une cle a la main ici.
    llm = RetryableChatOpenRouter(model=get_settings().OPENROUTER_AGENT_MODEL, temperature=0.1)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_ITERATIONS,
        early_stopping_method="force",
        verbose=True,
        handle_parsing_errors=True,
    )


def _load_local_seed_profiles(existing_urls: Set[str]) -> List[Dict[str, Any]]:
    seed_path = os.path.join("data", "talents_seed.json")
    if not os.path.exists(seed_path):
        return []

    with open(seed_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    profiles = []
    for item in payload or []:
        if not isinstance(item, dict):
            continue
        title = (item.get("title") or "").strip()
        url = (item.get("source_url") or "").strip()
        if not title:
            continue
        if url and url in existing_urls:
            continue
        profiles.append({
            "title": title,
            "description": item.get("description", ""),
            "source": "seed",
            "url": url or f"seed:{title}",
            "sector": item.get("sector", "other"),
            "country": item.get("country", "Maroc"),
            "expertise_tags": item.get("expertise_tags", ""),
            "years_experience": item.get("years_experience"),
            "image_url": "",
        })
    return profiles


def run_talent_scout_agent(
    existing_urls: Set[str] | None = None,
    existing_titles: Set[str] | None = None,
) -> List[Dict[str, Any]]:
    """
    Lance un cycle de decouverte pilote par l'agent : le LLM choisit lui-meme
    ses outils/sources et decide quand s'arreter (borne par MAX_ITERATIONS et
    MAX_VALIDATIONS).

    Retourne une liste de profils avec les memes cles que l'ancien pipeline
    (title, description, source, url, sector, country, expertise_tags,
    years_experience, image_url), pour rester compatible avec
    `discover_talents_from_sources` / talent_sync.py.
    """
    existing_urls = existing_urls or set()
    known_normalized_names = {_normalize_name(t) for t in (existing_titles or set())}
    discovered: List[Dict[str, Any]] = []
    raw_cache: Dict[str, Dict[str, Any]] = {}

    tools = _build_tools(existing_urls, known_normalized_names, discovered, raw_cache)

    try:
        executor = _build_agent_executor(tools)
    except Exception:
        logger.exception("Impossible de construire l'agent de decouverte de talents.")
        return []

    task = (
        "Trouve de nouveaux talents MRE en explorant plusieurs secteurs et sources "
        "differentes, verifie les doublons, et valide les candidats serieux."
    )

    try:
        executor.invoke({"input": task})
    except Exception:
        logger.exception("Erreur durant l'execution de l'agent de decouverte de talents.")

    if not discovered:
        search_tools = {getattr(tool, "name", ""): tool for tool in tools}
        try:
            sector_tool = search_tools.get("search_sector")
            if sector_tool:
                for domain in DOMAIN_QUERIES.keys():
                    try:
                        sector_tool.invoke(domain)
                    except Exception:
                        logger.exception("Fallback search_sector error pour le domaine '%s'", domain)
            for name in ("search_news", "search_rss", "search_orcid"):
                tool = search_tools.get(name)
                if tool:
                    try:
                        tool.invoke("")
                    except Exception:
                        logger.exception("Fallback %s error", name)
        except Exception:
            logger.exception("Fallback de collecte direct impossible")

        validate_tool = search_tools.get("validate_and_store_profile")
        check_tool = search_tools.get("check_duplicate")
        if validate_tool and check_tool:
            for url, entry in list(raw_cache.items()):
                if len(discovered) >= MAX_VALIDATIONS:
                    break
                title = entry.get("title", "")
                if not title:
                    continue
                try:
                    duplicate_message = check_tool.invoke(title)
                except Exception:
                    logger.exception("Fallback check_duplicate error pour '%s'", title)
                    duplicate_message = ""
                if "Doublon probable" in duplicate_message:
                    continue
                try:
                    validate_tool.invoke(url)
                except Exception:
                    logger.exception("Fallback validate_and_store_profile error pour '%s'", url)
                    continue

        if not discovered:
            seed_profiles = _load_local_seed_profiles(existing_urls)
            for profile in seed_profiles:
                title = profile.get("title", "")
                if not title:
                    continue
                if _is_likely_duplicate(title, known_normalized_names):
                    continue
                discovered.append(profile)

    logger.info(
        "Agent scout : %s profil(s) valide(s) sur %s piste(s) examinees.",
        len(discovered), len(raw_cache),
    )
    return discovered