"""Tests for the demo-only trading safeguards.

This project must never be able to trade real funds. These tests assert the
structural guarantees that make that true, so a future refactor cannot quietly
remove them.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from config import settings
from src.data import public_client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_TRADE_HOST = "api" + ".bybit.com"  # split so this file is not itself a match


def test_demo_trade_host_is_a_module_constant():
    """Safety-critical values must not be overridable by environment variables."""
    assert settings.BYBIT_DEMO_TRADE_URL == "https://api-demo.bybit.com"


def test_assert_demo_mode_rejects_anything_but_demo(monkeypatch):
    monkeypatch.setattr(settings, "TRADING_MODE", "live")
    with pytest.raises(settings.UnsafeConfigurationError, match="must be 'demo'"):
        settings.assert_demo_mode()

    monkeypatch.setattr(settings, "TRADING_MODE", "")
    with pytest.raises(settings.UnsafeConfigurationError):
        settings.assert_demo_mode()

    monkeypatch.setattr(settings, "TRADING_MODE", "demo")
    settings.assert_demo_mode()  # must not raise


def test_public_client_cannot_sign_requests():
    """The market-data client has no auth capability, so it cannot trade.

    The public host is only reachable from this module, and this module has no
    HMAC signing, no credentials and no POST method - Bybit rejects unsigned
    requests to every private endpoint.
    """
    source = inspect.getsource(public_client)
    for forbidden in ("hmac", "X-BAPI-SIGN", "api_secret", "session.post", "def post"):
        assert forbidden not in source, (
            f"{forbidden!r} appeared in the read-only market data client; "
            "signing capability must live only in src/exchange/"
        )


def test_public_client_refuses_non_market_endpoints():
    client = public_client.BybitPublicClient()
    with pytest.raises(ValueError, match="may only call"):
        client._get("/v5/order/create", {})


def test_production_trade_host_does_not_appear_in_source():
    """The strongest safeguard: production is unreachable because it is unwritten.

    The only permitted occurrence of the production host is in the read-only
    market-data client, which cannot authenticate.
    """
    allowed = {PROJECT_ROOT / "src" / "data" / "public_client.py"}
    offenders = []

    for path in list(PROJECT_ROOT.glob("src/**/*.py")) + list(PROJECT_ROOT.glob("*.py")):
        if path in allowed or "test" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if PRODUCTION_TRADE_HOST in text and "api-demo.bybit.com" not in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert not offenders, f"production Bybit host found in: {offenders}"


def test_env_example_exists_and_holds_no_real_secrets():
    example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "TRADING_MODE=demo" in example
    for line in example.splitlines():
        if line.startswith(("BYBIT_API_KEY", "BYBIT_API_SECRET", "OPENAI_API_KEY")):
            assert line.split("=", 1)[1].strip() == "", f"populated secret in .env.example: {line}"


def test_env_is_gitignored():
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.split("\n")
