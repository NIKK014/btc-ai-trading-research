"""Signed Bybit V5 client - demo trading only.

Safety architecture
-------------------
Four independent layers, any one of which prevents a real-money trade:

1. **The production host is not in this codebase.** This module imports
   :data:`~config.settings.BYBIT_DEMO_TRADE_URL` and has no other host. There
   is no string a typo could turn into the production endpoint, and a test
   enforces that no module outside the read-only market-data client mentions
   one.
2. **The host is a constant, not an environment variable.** A malformed
   ``.env`` cannot redirect order flow.
3. **:func:`assert_demo_mode` runs in the constructor.** The client cannot be
   built unless ``TRADING_MODE=demo``.
4. **Every order re-checks the base URL immediately before sending.** Cheap,
   and it survives refactors that move code around.

On top of that, the credentials themselves are demo-account keys, which Bybit
does not accept on the production endpoint at all. Two independent systems
would both have to fail.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import requests

from config.settings import BYBIT_DEMO_TRADE_URL, assert_demo_mode
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

RECV_WINDOW = "5000"


class BybitDemoError(RuntimeError):
    """The exchange rejected a request."""


class UnsafeEndpointError(RuntimeError):
    """A request was about to leave for somewhere other than the demo host."""


class BybitDemoClient:
    """Authenticated client for Bybit Demo Trading.

    Args:
        api_key: Demo API key. Falls back to ``BYBIT_API_KEY``.
        api_secret: Demo API secret. Falls back to ``BYBIT_API_SECRET``.
        session: Injectable for testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        session: Any = None,
        timeout: int = 15,
    ) -> None:
        # Layer 3: refuse to exist outside demo mode.
        assert_demo_mode()

        self.base_url = BYBIT_DEMO_TRADE_URL
        self.api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")
        self.timeout = timeout
        self._session = session or requests.Session()

        if not self.api_key or not self.api_secret:
            raise BybitDemoError(
                "BYBIT_API_KEY and BYBIT_API_SECRET must be set. "
                "Generate them from the Bybit DEMO account, not your real one."
            )

        logger.info("Bybit client ready | MODE: DEMO | host: %s", self.base_url)

    # -- safety ------------------------------------------------------------

    def _assert_demo_endpoint(self) -> None:
        """Layer 4: verified immediately before every request leaves."""
        if self.base_url != BYBIT_DEMO_TRADE_URL:
            raise UnsafeEndpointError(
                f"Refusing to send: base URL is {self.base_url!r}, "
                f"expected {BYBIT_DEMO_TRADE_URL!r}. This project may never "
                "trade real funds."
            )

    # -- signing -----------------------------------------------------------

    def _sign(self, timestamp: str, payload: str) -> str:
        """Bybit V5 HMAC: timestamp + api_key + recv_window + payload."""
        message = f"{timestamp}{self.api_key}{RECV_WINDOW}{payload}"
        return hmac.new(
            self.api_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _headers(self, payload: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "X-BAPI-SIGN": self._sign(timestamp, payload),
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        self._assert_demo_endpoint()
        url = f"{self.base_url}{path}"

        if method == "GET":
            query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            response = self._session.get(
                url, params=params, headers=self._headers(query), timeout=self.timeout
            )
        else:
            body = json.dumps(params, separators=(",", ":"))
            response = self._session.post(
                url, data=body, headers=self._headers(body), timeout=self.timeout
            )

        response.raise_for_status()
        payload = response.json()
        if payload.get("retCode") != 0:
            raise BybitDemoError(
                f"Bybit {payload.get('retCode')}: {payload.get('retMsg')} "
                f"({method} {path} {params})"
            )
        return payload.get("result", {})

    # -- account -----------------------------------------------------------

    def wallet_equity(self, coin: str = "USDT") -> float:
        """Total equity of the demo unified account."""
        result = self._request(
            "GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}
        )
        accounts = result.get("list", [])
        if not accounts:
            raise BybitDemoError("No wallet returned")

        account = accounts[0]
        for entry in account.get("coin", []):
            if entry.get("coin") == coin:
                for field in ("equity", "walletBalance"):
                    value = entry.get(field)
                    if value not in (None, ""):
                        return float(value)
        return float(account.get("totalEquity") or 0.0)

    def position(self, symbol: str, category: str = "linear") -> Dict[str, Any]:
        """Current position: ``{direction, size, entry_price, unrealised_pnl}``."""
        result = self._request(
            "GET", "/v5/position/list", {"category": category, "symbol": symbol}
        )
        entries = result.get("list", [])
        if not entries:
            return {"direction": 0, "size": 0.0, "entry_price": 0.0, "unrealised_pnl": 0.0}

        entry = entries[0]
        size = float(entry.get("size") or 0.0)
        side = entry.get("side", "")
        direction = 1 if side == "Buy" else (-1 if side == "Sell" else 0)
        return {
            "direction": direction if size > 0 else 0,
            "size": size,
            "entry_price": float(entry.get("avgPrice") or 0.0),
            "unrealised_pnl": float(entry.get("unrealisedPnl") or 0.0),
        }

    def set_leverage(self, symbol: str, leverage: float, category: str = "linear") -> None:
        """Set leverage, tolerating the "already set" response."""
        try:
            self._request(
                "POST",
                "/v5/position/set-leverage",
                {
                    "category": category,
                    "symbol": symbol,
                    "buyLeverage": str(leverage),
                    "sellLeverage": str(leverage),
                },
            )
            logger.info("Leverage set to %sx on %s", leverage, symbol)
        except BybitDemoError as exc:
            if "110043" in str(exc) or "not modified" in str(exc).lower():
                return  # already at this leverage
            raise

    # -- orders ------------------------------------------------------------

    def place_market_order(
        self,
        symbol: str,
        direction: int,
        quantity: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False,
        category: str = "linear",
        quantity_decimals: int = 3,
    ) -> str:
        """Submit a market order to the DEMO account. Returns the order id."""
        if direction == 0:
            raise ValueError("direction must be +1 or -1")

        quantity_string = f"{quantity:.{quantity_decimals}f}"
        if float(quantity_string) <= 0:
            raise ValueError(f"Quantity rounds to zero: {quantity}")

        params: Dict[str, Any] = {
            "category": category,
            "symbol": symbol,
            "side": "Buy" if direction > 0 else "Sell",
            "orderType": "Market",
            "qty": quantity_string,
            "timeInForce": "IOC",
        }
        if reduce_only:
            params["reduceOnly"] = True
        else:
            # Attach protective orders at submission, so the position is never
            # unprotected - not even for the seconds between two API calls.
            if stop_loss is not None:
                params["stopLoss"] = f"{stop_loss:.2f}"
            if take_profit is not None:
                params["takeProfit"] = f"{take_profit:.2f}"

        result = self._request("POST", "/v5/order/create", params)
        order_id = result.get("orderId", "")
        logger.info(
            "DEMO order placed: %s %s %s (sl=%s tp=%s) -> %s",
            params["side"],
            quantity_string,
            symbol,
            stop_loss,
            take_profit,
            order_id,
        )
        return order_id

    def close_position(self, symbol: str, category: str = "linear") -> Optional[str]:
        """Flatten any open position with a reduce-only market order."""
        current = self.position(symbol, category)
        if current["direction"] == 0 or current["size"] <= 0:
            return None
        return self.place_market_order(
            symbol=symbol,
            direction=-current["direction"],
            quantity=current["size"],
            reduce_only=True,
            category=category,
        )
