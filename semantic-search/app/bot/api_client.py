"""Async HTTP client for the QA API. The bot talks to the engine only through this."""

import httpx


class ApiUnavailable(Exception):
    """The API is down, timed out or returned a server error."""


class ApiRejected(Exception):
    """The API rejected the request (validation error, unknown id)."""


class QAClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = await self._http.request(method, path, **kwargs)
        except httpx.HTTPError as e:  # connection refused, timeout, ...
            raise ApiUnavailable(str(e) or type(e).__name__) from e
        if response.status_code >= 500:
            raise ApiUnavailable(f"{response.status_code} from {path}")
        if response.status_code >= 400:
            raise ApiRejected(response.json().get("detail", response.text))
        return response.json()

    async def ask(self, question: str, top_k: int = 3) -> dict:
        return await self._request(
            "POST", "/ask", json={"question": question, "top_k": top_k, "source": "bot"}
        )

    async def answer(self, answer_id: int) -> dict:
        return await self._request("GET", f"/answers/{answer_id}")

    async def feedback(self, query: str, answer_id: int | None, helpful: bool) -> dict:
        payload = {"query": query, "answer_id": answer_id, "helpful": helpful, "source": "bot"}
        return await self._request("POST", "/feedback", json=payload)

    async def stats(self) -> dict:
        return await self._request("GET", "/stats")

    async def aclose(self) -> None:
        await self._http.aclose()
