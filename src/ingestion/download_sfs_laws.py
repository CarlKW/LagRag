import requests
import json
import time
from datetime import datetime

OUTPUT_FILE = "sfs_lagboken_1990plus.jsonl"

START_YEAR = 1990
CURRENT_YEAR = datetime.today().year


def normalize_url(url: str) -> str:
    if not url:
        return url
    url = url.strip()

    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    elif url.startswith("https://"):
        pass
    elif url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = "https://data.riksdagen.se" + url
    else:
        if not url.startswith("https://data.riksdagen.se"):
            url = "https://data.riksdagen.se" + ("" if url.startswith("/") else "/") + url

    return url


def get_text_url(doc_meta: dict) -> str | None:
    url = doc_meta.get("dokument_url_text") or doc_meta.get("dokument_url_html")
    if not url:
        return None
    return normalize_url(url)


def fetch_fulltext(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def fetch_doclist_page(url: str, label: str, max_retries: int = 3):
    """
    Hämtar en dokumentlista-sida med retries.
    `label` används bara för loggen (t.ex. '1993 s.2').
    """
    url = normalize_url(url)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Hämtar {label}, försök {attempt}: {url}")
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"⚠️ Fel vid hämtning av {label}: {e}")
            if attempt < max_retries:
                time.sleep(5)
            else:
                print(f"❌ Ger upp {label} efter {max_retries} försök.")
                return None


def main():
    total_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for year in range(START_YEAR, CURRENT_YEAR + 1):
            from_date = f"{year}-01-01"
            if year < CURRENT_YEAR:
                tom_date = f"{year}-12-31"
            else:
                tom_date = ""

            print(f"\n==================== ÅR {year} ====================")

            if tom_date:
                current_url = (
                    "https://data.riksdagen.se/dokumentlista/"
                    f"?sok=&doktyp=sfs&rm=&from={from_date}&tom={tom_date}&ts=&bet=&tempbet=&nr=&org=&iid="
                    "&avd=&webbtv=&talare=&exakt=&planering=&sort=datum&sortorder=asc"
                    "&rapport=&utformat=json&a=s"
                )
            else:
                current_url = (
                    "https://data.riksdagen.se/dokumentlista/"
                    f"?sok=&doktyp=sfs&rm=&from={from_date}&tom=&ts=&bet=&tempbet=&nr=&org=&iid="
                    "&avd=&webbtv=&talare=&exakt=&planering=&sort=datum&sortorder=asc"
                    "&rapport=&utformat=json&a=s"
                )

            page_no = 1
            year_count = 0

            while True:
                label = f"år {year}, sida {page_no}"
                data = fetch_doclist_page(current_url, label)
                if data is None:
                    break

                doc_list = data.get("dokumentlista", {})
                docs = doc_list.get("dokument", [])

                if not docs:
                    print(f"Inga dokument på {label} — bryter året {year}.")
                    break

                for d in docs:
                    dok_id = d.get("dok_id")
                    titel = d.get("titel")
                    beteckning = d.get("beteckning")
                    datum = d.get("datum")

                    text_url = get_text_url(d)
                    if not text_url:
                        print(f"⚠️ Ingen text-URL för {dok_id} ({titel}), hoppar över.")
                        continue

                    try:
                        fulltext = fetch_fulltext(text_url)
                    except requests.HTTPError as e:
                        if e.response is not None and e.response.status_code == 404:
                            print(f"⚠️ 404 för {dok_id} ({titel}), hoppar över.")
                            continue
                        print(f"⚠️ HTTP-fel för {dok_id} ({titel}): {e}")
                        continue
                    except Exception as e:
                        print(f"⚠️ Kunde inte hämta text för {dok_id} ({titel}): {e}")
                        continue

                    record = {
                        "dok_id": dok_id,
                        "titel": titel,
                        "beteckning": beteckning,
                        "datum": datum,
                        "doktyp": d.get("doktyp"),
                        "rm": d.get("rm"),
                        "metadata": d,
                        "fulltext": fulltext,
                    }

                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_count += 1
                    year_count += 1

                    # Var snäll mot servern
                    time.sleep(0.05)

                print(f"År {year}, sida {page_no} klar, {year_count} dokument för året, {total_count} totalt.")

                next_url = doc_list.get("@nasta_sida")
                if next_url:
                    current_url = next_url
                    page_no += 1
                    if page_no > 10000:  # skydd mot ev. buggar
                        print(f"För många sidor för år {year} – bryter.")
                        break
                else:
                    print(f"Ingen @nasta_sida för år {year} — klart för året.")
                    break

            time.sleep(2)

    print(f"\n✅ Klar! Sparade totalt {total_count} SFS-dokument (från {START_YEAR}) i {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
