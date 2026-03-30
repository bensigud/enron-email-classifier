import re
import email
import signal
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count


def _timeout_handler(signum, frame):
    raise TimeoutError("File parse timed out")


def _parse_file(file_path: Path) -> dict:
    """Parse a single email file. Runs in parallel across CPU cores."""
    try:
        # Kill any file that takes more than 5 seconds (handles corrupted files)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(5)

        raw = file_path.read_text(errors="ignore")
        msg = email.message_from_string(raw)

        sender = msg.get("From", "").strip()
        if not sender:
            signal.alarm(0)
            return None

        body = _extract_body(msg)
        if not body:
            signal.alarm(0)
            return None

        result = {
            "message_id": msg.get("Message-ID", str(file_path)).strip(),
            "sender":     _clean_email(sender),
            "recipients": _parse_recipients(msg.get("To", "")),
            "date":       msg.get("Date", ""),
            "subject":    msg.get("Subject", "").strip(),
            "body":       clean_text(body),
        }
        signal.alarm(0)
        return result
    except Exception:
        signal.alarm(0)
        return None


def load_emails(data_path: str, max_emails: int = None) -> pd.DataFrame:
    """
    Load all Enron emails using multiple CPU cores in parallel.
    Much faster than reading files one by one.
    """
    data_path = Path(data_path)
    all_files = [f for f in data_path.rglob("*") if f.is_file()]

    if max_emails:
        all_files = all_files[:max_emails]

    total = len(all_files)
    cores = cpu_count()
    print(f"Loading {total:,} files using {cores} CPU cores...")
    print(f"Progress will update every 1,000 files.\n")

    records = []
    done = 0

    with Pool(processes=cores) as pool:
        for result in pool.imap_unordered(_parse_file, all_files, chunksize=50):
            done += 1
            if result is not None:
                records.append(result)

            if done % 1000 == 0 or done == total:
                pct = done / total * 100
                print(f"  [{pct:5.1f}%] {done:,} / {total:,} files processed — {len(records):,} valid emails so far", flush=True)

    df = pd.DataFrame(records)

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)

    print(f"\nDone! Loaded {len(df):,} emails")
    return df


def _extract_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                body = part.get_payload(decode=True)
                if body:
                    return body.decode("utf-8", errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            return payload.decode("utf-8", errors="ignore")
    return ""


def _clean_email(address: str) -> str:
    match = re.search(r"[\w\.-]+@[\w\.-]+", address)
    return match.group(0).lower() if match else address.lower().strip()


def _parse_recipients(recipients_str: str) -> list:
    if not recipients_str:
        return []
    parts = re.split(r"[,;]", recipients_str)
    return [_clean_email(p) for p in parts if p.strip()]


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"-{3,}.*?(Original Message|Forwarded by).*?\n", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"^(From|To|Cc|Subject|Date|Sent):.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def save_processed(df: pd.DataFrame, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} emails to {output_path}")


def load_processed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["recipients"] = df["recipients"].apply(eval)
    print(f"Loaded {len(df):,} emails from {path}")
    return df
