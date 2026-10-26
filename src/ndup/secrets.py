"""Secret lookup, log redaction, and Notion token rotation.

Lookup order: ``/run/secrets/<name>``, then the environment, then the OS keyring,
then ``~/.config/ndup/secrets/<name>``. The RunPod key is ``RUNPOD_API_KEY``.
Secret values are never written to config, logs, or exports.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path

from ndup.errors import NdUpError

SECRET_ENV: dict[str, str] = {
    "notion_token": "NDUP_NOTION_TOKEN",
    "notion_webhook_token": "NDUP_NOTION_WEBHOOK_TOKEN",
    "openai_api_key": "NDUP_OPENAI_API_KEY",
    "pg_password_worker": "NDUP_PG_PASSWORD_WORKER",
    "pg_password_web": "NDUP_PG_PASSWORD_WEB",
    "proxy_shared_secret": "NDUP_PROXY_SHARED_SECRET",
    "cloudflare_tunnel_token": "NDUP_CLOUDFLARE_TUNNEL_TOKEN",
    "backup_credentials": "NDUP_BACKUP_CREDENTIALS",
    "text_encryption_key": "NDUP_TEXT_ENCRYPTION_KEY",
    "runpod_api_key": "RUNPOD_API_KEY",
}

_INSTALLED: list[str] = []


class SecretError(NdUpError):
    """A secret could not be read or rotated."""


def resolve_secret(name: str, secrets_dir: Path | None = None) -> str | None:
    """Return the secret value, or None when it is not configured."""
    directory = secrets_dir if secrets_dir is not None else Path("/run/secrets")
    file_path = directory / name
    if file_path.is_file():
        value = file_path.read_text(encoding="utf-8").strip()
        if value:
            return value
    env_name = SECRET_ENV.get(name)
    if env_name:
        env_value = os.environ.get(env_name)
        if env_value:
            return env_value
    keyring_value = _read_keyring(name)
    if keyring_value:
        return keyring_value
    user_file = user_secret_path(name)
    if user_file.is_file():
        value = user_file.read_text(encoding="utf-8").strip()
        if value:
            return value
    return None


def user_secret_path(name: str) -> Path:
    """Mode-600 fallback outside the repo when the OS keyring is unavailable."""
    return Path.home() / ".config" / "ndup" / "secrets" / name


def store_secret(name: str, value: str) -> str:
    """Store a secret in the OS keyring, or a mode-600 file outside the repo.

    The return value is ``keyring`` or a filesystem path. It never includes
    the secret itself.
    """
    cleaned = value.strip()
    if not cleaned:
        raise SecretError(f"Refusing to store an empty {name}")
    if _write_keyring(name, cleaned):
        return "keyring"
    target = user_secret_path(name)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(cleaned + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
    return str(target)


def known_secret_values(secrets_dir: Path | None = None) -> list[str]:
    found: list[str] = []
    for name in SECRET_ENV:
        value = resolve_secret(name, secrets_dir=secrets_dir)
        if value:
            found.append(value)
    return found


def _read_keyring(name: str) -> str | None:
    try:
        import keyring
    except ImportError:
        return None
    try:
        value = keyring.get_password("ndup", name)
    except Exception:
        return None
    if isinstance(value, str) and value:
        return value
    return None


def _write_keyring(name: str, value: str) -> bool:
    try:
        import keyring
    except ImportError:
        return False
    try:
        keyring.set_password("ndup", name, value)
    except Exception:
        return False
    return True


class RedactFilter(logging.Filter):
    """Replace known secret values in log records."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        self.secrets = [item for item in secrets if item]

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(str(record.msg))
        if isinstance(record.args, dict):
            record.args = {key: self._redact(str(value)) for key, value in record.args.items()}
        elif isinstance(record.args, tuple):
            record.args = tuple(self._redact(str(value)) for value in record.args)
        return True

    def _redact(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "***")
        return text


def install_redaction(secrets: list[str] | None = None) -> RedactFilter:
    """Attach a redaction filter to the ndup logger and the root logger."""
    values = list(secrets) if secrets is not None else known_secret_values()
    redactor = RedactFilter(values)
    _INSTALLED[:] = values
    logger = logging.getLogger("ndup")
    logger.addFilter(redactor)
    logging.getLogger().addFilter(redactor)
    return redactor


def rotate_notion_token(
    new_token: str,
    validate: Callable[[str], bool],
    secrets_dir: Path | None = None,
) -> Path:
    """Validate a replacement Notion token and store it.

    Docker secret files win when the directory is writable. Otherwise the token
    is written to ``~/.config/ndup/secrets/notion_token`` with mode 600.
    """
    if not new_token or not new_token.strip():
        raise SecretError("Refusing to store an empty Notion token")
    if not validate(new_token):
        raise SecretError("Notion rejected the new token")
    directory = secrets_dir if secrets_dir is not None else Path("/run/secrets")
    if directory.is_dir() and os.access(directory, os.W_OK):
        target = directory / "notion_token"
    else:
        target = Path.home() / ".config" / "ndup" / "secrets" / "notion_token"
        target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(new_token.strip() + "\n", encoding="utf-8")
    os.chmod(target, 0o600)
    return target
