#!/usr/bin/env python3
import argparse
import os
import sys
import time
import re

from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

BASE_URL = "https://www.theyworkforyou.com/pwdata/scrapedxml/debates/"

def make_session():
    s = requests.Session()
    retries = Retry(
        total=5, backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    s.headers.update({"User-Agent": "IR Coursework Downloader (educational use)"})
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    return s

def list_remote_files(session, base_url: str):
    # קורא את הדף כטקסט פשוט ומחלץ href-ים עם regex (בלי BeautifulSoup)
    r = session.get(base_url, timeout=60)
    r.raise_for_status()
    # מחלץ רק קבצים שמתחילים ב-debates ומסתיימים ב-xml
    files = set(re.findall(r'href="(debates[^"]+?\.xml)"', r.text))
    # מיון לקסיקוגרפי (פורמט debatesYYYY-MM-DD[a-d].xml)
    return sorted(files)

def download_one(session, base_url: str, fname: str, out_dir: str):
    url = urljoin(base_url, fname)
    out_path = os.path.join(out_dir, fname)
    # דילוג אם כבר יורד/ירד
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return "skipped"
    with session.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        tmp_path = out_path + ".part"
        total = int(r.headers.get("Content-Length", 0))
        with open(tmp_path, "wb") as f, tqdm(
            total=total if total > 0 else None,
            unit="B", unit_scale=True, leave=False, desc=fname
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                if chunk:
                    f.write(chunk)
                    if total > 0:
                        pbar.update(len(chunk))
        os.replace(tmp_path, out_path)
    return "downloaded"

def main():
    ap = argparse.ArgumentParser(description="Download UK Hansard debates XML (from a given filename onwards).")
    ap.add_argument("--since", default="debates2023-06-28d.xml",
                    help="התחילי מקובץ זה (לקסיקוגרפית כולל הקובץ). ברירת מחדל: debates2023-06-28d.xml")
    ap.add_argument("--out", default="debates_xml",
                    help="תיקיית פלט להורדה (תיווצר אם לא קיימת).")
    ap.add_argument("--base-url", default=BASE_URL,
                    help="כתובת הבסיס של התיקייה (למקרה שתרצי לשנות).")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    session = make_session()

    try:
        all_files = list_remote_files(session, args.base_url)
    except Exception as e:
        print(f" נכשל קריאת אינדקס: {e}", file=sys.stderr)
        sys.exit(1)

    # סינון מהקובץ המבוקש והלאה
    to_get = [f for f in all_files if f >= args.since]
    if not to_get:
        print("אין קבצים להורדה (בדקי את --since).")
        return

    print(f"יימצאו {len(to_get)} קבצים להורדה (מתוך {len(all_files)}).")
    ok = 0; skip = 0; fail = 0
    for fname in to_get:
        try:
            status = download_one(session, args.base_url, fname, args.out)
            if status == "downloaded":
                ok += 1
            else:
                skip += 1
        except Exception as e:
            fail += 1
            print(f"⚠️  כשל בקובץ {fname}: {e}", file=sys.stderr)
            time.sleep(1)

    print(f"\nסיכום: הורדו {ok}, דולגו {skip}, נכשלו {fail}. קבצים בתיקייה: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
