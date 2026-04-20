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
            "cc":         _parse_recipients(msg.get("Cc", "") or msg.get("CC", "")),
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

    # Filter junk emails (auto-replies, newsletters, data dumps)
    before = len(df)
    df = df[~df.apply(lambda r: is_junk_email(r["subject"], r["body"]), axis=1)]
    removed = before - len(df)
    print(f"  Junk filter removed {removed:,} emails ({removed/before*100:.1f}%)")

    # Deduplicate near-identical emails
    before_dedup = len(df)
    df = deduplicate_emails(df)
    dedup_removed = before_dedup - len(df)
    print(f"  Dedup removed {dedup_removed:,} near-duplicate emails ({dedup_removed/max(before_dedup,1)*100:.1f}%)")

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
    """
    Clean an email body: remove quoted replies, forwarded content,
    headers, and URLs. Only keep the original message the sender wrote.
    """
    if not text:
        return ""

    # Cut everything after common reply/forward markers
    # These indicate the start of quoted previous messages
    cut_patterns = [
        r"-{3,}\s*Original Message\s*-{3,}",
        r"-{3,}\s*Forwarded by\s",
        r"_{3,}\s*\n",                        # _____ separator lines
        r"={3,}\s*\n",                        # ===== separator lines
        r"^On .+ wrote:\s*$",                 # "On Mon, Jan 1, X wrote:"
        r"^>.*$(\n^>.*$)+",                   # Lines starting with >
        r"^From:.*\nSent:.*\nTo:",            # Outlook-style quoted headers
    ]
    for pattern in cut_patterns:
        match = re.search(pattern, text, flags=re.MULTILINE | re.IGNORECASE)
        if match:
            text = text[:match.start()]

    # Remove stray header lines that survived
    text = re.sub(r"^(From|To|Cc|Subject|Date|Sent):.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # Remove URLs
    text = re.sub(r"http\S+", "", text)
    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def deduplicate_emails(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove near-duplicate emails. Two emails are duplicates if they
    have the same sender, same first recipient, and the first 100
    characters of the body match. Keeps the earliest copy.

    This catches the common Enron pattern where the same email appears
    in both the sender's "sent" folder and the recipient's inbox, or
    where a forwarded chain duplicates the original message.
    """
    df = df.sort_values("date").copy()

    # Build a fingerprint from sender + first recipient + body start
    def _fingerprint(row):
        sender = row["sender"] or ""
        recip = row["recipients"][0] if isinstance(row["recipients"], list) and row["recipients"] else ""
        body = (row["body"] or "")[:100].strip().lower()
        # Normalize whitespace for matching
        body = re.sub(r"\s+", " ", body)
        return f"{sender}|{recip}|{body}"

    df["_fingerprint"] = df.apply(_fingerprint, axis=1)
    df = df.drop_duplicates(subset="_fingerprint", keep="first")
    df = df.drop(columns="_fingerprint")

    return df


def is_junk_email(subject: str, body: str) -> bool:
    """
    Light spam/junk filter to remove emails with no relationship signal.
    Returns True if the email should be excluded.
    """
    subj = (subject or "").lower().strip()
    text = (body or "").strip()

    # Too short to carry any signal
    if len(text) < 30:
        return True

    # Automated system messages
    auto_patterns = [
        r"out of office",
        r"automatic reply",
        r"auto.?reply",
        r"delivery failure",
        r"undeliverable",
        r"returned mail",
        r"mail delivery",
        r"postmaster",
        r"do not reply",
        r"noreply",
        r"system administrator",
        r"virus alert",
        r"server notification",
    ]
    combined = subj + " " + text[:200].lower()
    for pattern in auto_patterns:
        if re.search(pattern, combined):
            return True

    # Newsletters / mass distribution
    newsletter_patterns = [
        r"unsubscribe",
        r"click here to",
        r"this message was sent to",
        r"to be removed from",
        r"mailing list",
        r"daily news",
        r"weekly report",
        r"energy bulletin",
    ]
    for pattern in newsletter_patterns:
        if re.search(pattern, text.lower()):
            return True

    # Body is mostly forwarded headers / no original content
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return True

    # Mostly non-alpha characters (tables, data dumps, legal text)
    alpha_chars = sum(1 for c in text if c.isalpha())
    if len(text) > 50 and alpha_chars / len(text) < 0.3:
        return True

    return False


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
    output_path = Path(output_path).with_suffix(".parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, engine="pyarrow")
    print(f"Saved {len(df):,} emails to {output_path} "
          f"({output_path.stat().st_size / 1024 / 1024:.1f} MB)")


def load_processed(path: str) -> pd.DataFrame:
    import time
    path = Path(path).with_suffix(".parquet")
    t0 = time.time()
    df = pd.read_parquet(path, engine="pyarrow")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    if "recipients" in df.columns and len(df) > 0:
        first = df["recipients"].iloc[0]
        if isinstance(first, str):
            df["recipients"] = df["recipients"].apply(ast.literal_eval)
    elapsed = time.time() - t0
    print(f"Loaded {len(df):,} emails from {path.name} ({elapsed:.1f}s)")
    return df
