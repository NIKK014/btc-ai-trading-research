"""Signed Bybit V5 client - Bybit Testnet only.

Safety architecture
-------------------
Four independent layers, any one of which prevents a real-money trade:

1. **The production host is not in this codebase.** This module imports
   :data:`~config.settings.BYBIT_PAPER_TRADE_URL` and has no other host. There
   is no string a typo could turn into the production endpoint, and a test
   enforces that no module outside the read-only market-data client mentions
   one.
2. **The host is a constant, not an environment variable.** A malformed
   ``.env`` cannot redirect order flow.
3. **:func:`assert_paper_mode` runs in the constructor.** The client cannot be
   built unless ``TRADING_MODE`` is a paper mode.
4. **Every order re-checks the base URL immediately before sending.** Cheap,
   and it survives refactors that move code around.

On top of that, testnet credentials are issued by an entirely separate system
and are not recognised on mainnet at all. Testnet balances are not
convertible to anything: there is no mechanism by which an order placed here
could touch real money.

Why testnet rather than Demo Trading: Bybit EU cannot offer perpetual futures
under MiCA, and this research requires the ability to go short. Testnet
provides linear perpetuals with faucet-funded balances.

Caveat worth stating in the write-up: testnet runs its own order book, so its
prices drift from the real market. Signals are computed from mainnet history
while orders execute against testnet's book, which means the live run
demonstrates that the pipeline works end to end - it does not produce
meaningful P&L.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from typing import Any, Dict, Optional

import requests

from config.settings import BYBIT_PAPER_TRADE_URL, assert_paper_mode
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

RECV_WINDOW = "5000"


class BybitPaperError(RuntimeError):
    """The exchange rejected a request."""


class UnsafeEndpointError(RuntimeError):
    """A request was about to leave for somewhere other than the paper host."""


class BybitPaperClient:
    """Authenticated client for Bybit Testnet.

    Args:
        api_key: Testnet API key. Falls back to ``BYBIT_API_KEY``.
        api_secret: Testnet API secret. Falls back to ``BYBIT_API_SECRET``.
        session: Injectable for testing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        session: Any = None,
        timeout: int = 15,
    ) -> None:
        # Layer 3: refuse to exist outside a paper-trading mode.
        assert_paper_mode()

        self.base_url = BYBIT_PAPER_TRADE_URL
        self.api_key = api_key or os.getenv("BYBIT_API_KEY", "")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET", "")
        self.timeout = timeout
        self._session = session or requests.Session()

        if not self.api_key or not self.api_secret:
            raise BybitPaperError(
                "BYBIT_API_KEY and BYBIT_API_SECRET must be set. "
                "Generate them at testnet.bybit.com, not on the live exchange."
            )

        logger.info("Bybit client ready | MODE: TESTNET (paper) | host: %s", self.base_url)

    # -- safety ------------------------------------------------------------

    def _assert_paper_endpoint(self) -> None:
        """Layer 4: verified immediately before every request leaves."""
        if self.base_url != BYBIT_PAPER_TRADE_URL:
            raise UnsafeEndpointError(
                f"Refusing to send: base URL is {self.base_url!r}, "
                f"expected {BYBIT_PAPER_TRADE_URL!r}. This project may never "
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
        self._assert_paper_endpoint()
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

        if response.status_code in (401, 403):
            raise BybitPaperError(
                f"Bybit rejected the credentials ({response.status_code}: "
                f"{response.text.strip()[:120]}).\n\n"
                "  Testnet uses SEPARATE API keys from any mainnet account.\n"
                "  A mainnet or Bybit EU key will always be rejected here - "
                "which is the safeguard working.\n\n"
                "  To create the right key:\n"
                "    1. Go to testnet.bybit.com and register (it is a separate\n"
                "       signup from bybit.com; no KYC, no real funds involved)\n"
                "    2. Fund it from the testnet faucet: Assets -> Request funds\n"
                "    3. Avatar menu -> API -> Create New Key (System-generated)\n"
                "    4. Permissions: Read-Write, and enable Contract Trade\n"
                "    5. Put it in .env as BYBIT_API_KEY / BYBIT_API_SECRET\n\n"
                "  If the key is definitely a testnet key, check your system "
                "clock: Bybit rejects requests whose timestamp is out of sync."
            )

        response.raise_for_status()
        payload = response.json()
        if payload.get("retCode") != 0:
            raise BybitPaperError(
                f"Bybit {payload.get('retCode')}: {payload.get('retMsg')} "
                f"({method} {path} {params})"
            )
        return payload.get("result", {})

    # -- account -----------------------------------------------------------

    def wallet_equity(self, coin: str = "USDT") -> float:
        """Total equity of the testnet unified account."""
        result = self._request(
            "GET", "/v5/account/wallet-balance", {"accountType": "UNIFIED"}
        )
        accounts = result.get("list", [])
        if not accounts:
            raise BybitPaperError("No wallet returned")

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

    def last_price(self, symbol: str, category: str = "linear") -> float:
        """Latest traded price on the venue we are actually trading on.

        Testnet runs its own order book, so its price can differ from the real
        market by a wide margin. Orders must be sized and stopped against the
        price they will actually fill at, not against the mainnet price the
        signal was derived from - otherwise stops land nowhere near the book
        and fill instantly or never.
        """
        result = self._request(
            "GET", "/v5/market/tickers", {"category": category, "symbol": symbol}
        )
        tickers = result.get("list", [])
        if not tickers:
            raise BybitPaperError(f"No ticker returned for {symbol}")
        return float(tickers[0]["lastPrice"])

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
        except BybitPaperError as exc:
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
        """Submit a market order to the TESTNET account. Returns the order id."""
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
            "TESTNET order placed: %s %s %s (sl=%s tp=%s) -> %s",
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
