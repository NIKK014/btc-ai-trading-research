"""Read-only Bybit market-data client.

This module talks to Bybit's public market-data host. It is deliberately kept
separate from ``src/exchange/`` and has three properties that make it
incapable of trading:

1. It never sends API credentials.
2. It contains no request-signing code, and Bybit rejects unsigned requests to
   any private endpoint.
3. It only ever issues GET requests to ``/v5/market/*``.

Order placement lives exclusively in ``src/exchange/``, which is hardwired to
the demo host. Keeping the two hosts in two modules with different
capabilities means no single typo can route an order to production.
"""

from __future__ import annotations

import time
from typing import Any

import requests

from config.settings import BYBIT_PUBLIC_DATA_URL, DATA
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

_ALLOWED_PATH_PREFIX = "/v5/market/"


class BybitAPIError(RuntimeError):
    """Bybit returned a non-zero ``retCode`` or the request kept failing."""


class BybitPublicClient:
    """Minimal, unauthenticated wrapper over Bybit's public market endpoints."""

    def __init__(
        self,
        base_url: str = BYBIT_PUBLIC_DATA_URL,
        timeout: int = DATA.request_timeout,
        max_retries: int = DATA.max_retries,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "btc-ai-trader-research/0.1"})

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """GET a public endpoint with exponential backoff on transient errors."""
        if not path.startswith(_ALLOWED_PATH_PREFIX):
            raise ValueError(
                f"BybitPublicClient may only call {_ALLOWED_PATH_PREFIX}* "
                f"endpoints (got {path!r})."
            )

        url = f"{self.base_url}{path}"
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                if payload.get("retCode") != 0:
                    raise BybitAPIError(
                        f"Bybit error {payload.get('retCode')}: "
                        f"{payload.get('retMsg')} (params={params})"
                    )
                return payload["result"]
            except (requests.RequestException, BybitAPIError, ValueError) as exc:
                last_error = exc
                backoff = 2**attempt
                logger.warning(
                    "Request failed (attempt %d/%d), retrying in %ds: %s",
                    attempt + 1,
                    self.max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)

        raise BybitAPIError(f"GET {url} failed after {self.max_retries} attempts") from last_error

    def get_kline(
        self,
        symbol: str,
        interval: str,
        category: str = "linear",
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1000,
    ) -> list[list[str]]:
        """Fetch raw klines.

        Returns Bybit's raw rows, newest first, each row being
        ``[start_time_ms, open, high, low, close, volume, turnover]``.

        ``start_time_ms`` is the candle's OPEN time. A candle is only complete
        at ``open_time + interval``; the caller is responsible for discarding
        any still-forming final candle.
        """
        params: dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_ms is not None:
            params["start"] = start_ms
        if end_ms is not None:
            params["end"] = end_ms

        result = self._get("/v5/market/kline", params)
        return result.get("list", [])

    def get_last_price(self, symbol: str, category: str = "linear") -> float:
        """Latest traded price, for dashboard display and live sizing."""
        result = self._get("/v5/market/tickers", {"category": category, "symbol": symbol})
        tickers = result.get("list", [])
        if not tickers:
            raise BybitAPIError(f"No ticker returned for {symbol}")
        return float(tickers[0]["lastPrice"])
