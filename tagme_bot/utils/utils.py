import asyncio
from typing import List, Dict, Any

from crowd_sdk.core.utils.http_client import APIResponseError
from crowd_sdk.tagme import TagmeClientAdvanced
import aiohttp
from bs4 import BeautifulSoup

from utils.decorators import Cache, handle_http_errors

cache = Cache()


@cache
async def get_method_and_task_files(client: TagmeClientAdvanced, task_id: str, org_id: str):
    try:
        td = await client.get_task(task_id=task_id, organization_id=org_id)
        pd = await client.get_project(project_id=td.project_id, organization_id=org_id)
        md = await client.get_method(method_id=pd.method_id, task_id=task_id, organization_id=org_id)
        tf = await client.get_task_files(task_id=task_id, organization_id=org_id)
        return td, md, tf
    except APIResponseError as e:
        print(f"  ️ Skipping task {task_id} (org={org_id}): {e}")
        return


@cache
async def download_instruction(url, token: str, org_id: str) -> str:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/plain",
        "Organization-Id": org_id,
    }
    timeout = aiohttp.ClientTimeout(total=60, connect=30, sock_read=30, sock_connect=30)
    data = b""
    for attempt in range(1, 4):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers, ssl=False) as resp:
                    resp.raise_for_status()
                    data = await resp.read()
            break
        except Exception:
            if attempt < 3:
                await asyncio.sleep(1 * attempt)
                continue
            else:
                return ""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("cp1251", errors="replace")
    return BeautifulSoup(text, "html.parser").get_text("\n").strip()
