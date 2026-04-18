import re
import ast
import email
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count


def _parse_file(file_path: Path) -> dict:
    """Parse a single email file. Runs in parallel across CPU cores."""
    try:
        raw = file_path.read_text(errors="ignore")
        msg = email.message_from_string(raw)

        sender = msg.get("From", "").strip()
        if not sender:
            return None

        body = _extract_body(msg)
        if not body:
            return None

        result = {
            "message_id": msg.get("Message-ID", str(file_path)).strip(),
            "sender":     _clean_email(sender),
            "recipients": _parse_recipients(msg.get("To", "")),
            "date":       msg.get("Date", ""),
            "subject":    msg.get("Subject", "").strip(),
            "body":       clean_text(body),
        }
        return result
    except Exception:
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


def get_executive_addresses(data_path: str) -> set:
    """
    Get the set of executive email addresses from the maildir folder names.
    Each folder in maildir/ is one executive's mailbox (e.g. 'lay-k', 'skilling-j').
    We convert folder names to the most common email pattern: first.last@enron.com
    """
    data_path = Path(data_path)
    if not data_path.exists():
        return set()

    folders = [f.name for f in data_path.iterdir() if f.is_dir()]

    # Build possible email patterns from folder names
    # Folder format is usually "lastname-f" (e.g. "lay-k", "skilling-j")
    addresses = set()
    for folder in folders:
        parts = folder.split("-")
        if len(parts) >= 2:
            lastname = parts[0]
            firstinit = parts[1]
            # Common Enron patterns
            addresses.add(f"{firstinit}.{lastname}@enron.com")
            addresses.add(f"{firstinit}{lastname}@enron.com")
            addresses.add(f"{lastname}.{firstinit}@enron.com")
            addresses.add(f"{lastname}{firstinit}@enron.com")
            addresses.add(f"{firstinit}.{lastname}@enron.com".lower())

    return addresses


def filter_executives_only(df: pd.DataFrame, data_path: str) -> pd.DataFrame:
    """
    Filter the DataFrame to only keep emails where BOTH the sender
    and at least one recipient are executives (have a mailbox in maildir).

    We identify executive email addresses by looking at emails in each
    person's "sent_items" or "sent" folder — the sender of those emails
    is that executive's actual email address.
    """
    data_path = Path(data_path)

    # Step 1: Find each executive's real email address from their sent folders
    exec_addresses = set()
    for folder in data_path.iterdir():
        if not folder.is_dir():
            continue
        # Look for sent mail folders — these contain emails FROM this executive
        sent_folders = []
        for subfolder in folder.iterdir():
            if subfolder.is_dir() and subfolder.name in ("sent", "sent_items", "_sent_mail"):
                sent_folders.append(subfolder)

        # Get the sender address from emails in their sent folder
        for sent_folder in sent_folders:
            sent_files = [f for f in sent_folder.rglob("*") if f.is_file()]
            for sent_file in sent_files[:5]:  # only need a few to find the address
                try:
                    raw = sent_file.read_text(errors="ignore")
                    msg = email.message_from_string(raw)
                    sender = msg.get("From", "").strip()
                    if sender:
                        addr = _clean_email(sender)
                        if addr and "@" in addr:
                            exec_addresses.add(addr)
                            break
                except Exception:
                    continue

    # Also include addresses derived from folder names as fallback
    exec_addresses.update(get_executive_addresses(str(data_path)))

    print(f"  Found {len(exec_addresses):,} executive email addresses")

    # Filter: sender must be an executive
    df = df[df["sender"].isin(exec_addresses)].copy()

    # Filter: at least one recipient must be an executive
    df["recipients"] = df["recipients"].apply(
        lambda recips: [r for r in recips if r in exec_addresses]
    )
    df = df[df["recipients"].apply(len) > 0]

    print(f"  After filtering: {len(df):,} emails between executives")
    return df


def save_processed(df: pd.DataFrame, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} emails to {output_path}")


def load_processed(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df["recipients"] = df["recipients"].apply(ast.literal_eval)
    print(f"Loaded {len(df):,} emails from {path}")
    return df
