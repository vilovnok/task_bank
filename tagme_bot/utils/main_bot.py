import asyncio
import re

import aiohttp
from bs4 import BeautifulSoup


class MainBot:

    def split_deepseek(self, raw: str):
        if "<think>" in raw and "</think>" in raw:
            head, tail = re.split(r"</think>\s*", raw, maxsplit=1)
            think = re.sub(r"\A\s*<think>\s*", "", head, flags=re.DOTALL).strip()
            return think, tail.strip()
        if raw.lstrip().startswith("<think>"):
            return raw.split("<think>", 1)[1].strip(), ""
        return "", raw.strip()

    async def download_instruction(self, url: str, token: str, org_id: str) -> str:
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "text/plain",
            "Organization-Id": org_id,
        }
        timeout = aiohttp.ClientTimeout(total=60, connect=30, sock_read=30, sock_connect=30)
        data = b""
        for attempt in range(1, 4):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as sess:
                    async with sess.get(url, headers=headers, ssl=False) as resp:
                        resp.raise_for_status()
                        data = await resp.read()
                break
            except Exception as e:
                print(f"[download_instruction] попытка {attempt} ошибка: {e}")
                if attempt < 3:
                    await asyncio.sleep(attempt)
                else:
                    return ""
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("cp1251", errors="replace")
        return BeautifulSoup(text, "html.parser").get_text("\n").strip()