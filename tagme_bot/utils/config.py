from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional
import os

CRED_ENV_BY_STAND = {
    "cloud":  ("USERNAME_DE",    "PASSWORD_DE"),
    "sigma":  ("USERNAME_SIGMA", "PASSWORD_SIGMA"),
    "sberdev": ("USERNAME_SD",   "PASSWORD_SD"),
}

@dataclass(frozen=True)
class StandConfig:
    name: str
    host: str
    cfg_path: str
    seen_file: str
    organizations: List[str]
    skip_project_names: List[str]
    require_service: bool = False
    forbidden_services: FrozenSet[str] = frozenset()


def check_service_policy(stand: "StandConfig", service: Optional[str]) -> Optional[str]:
    """Возвращает текст ошибки, если service запрещён политикой стенда, иначе None."""
    if stand.require_service and not service:
        return f"Для {stand.name} необходимо указать service"
    if service and service in stand.forbidden_services:
        return f"Для {stand.name} запрещено использование service {service}"
    return None


def _expand(p: str) -> str:
    return os.path.expanduser(p)

def get_creds_for_stand(stand_name: str) -> tuple[str, str]:
    u_key, p_key = CRED_ENV_BY_STAND[stand_name]
    username = os.getenv(u_key)
    password = os.getenv(p_key)
    if not username or not password:
        raise RuntimeError(f"Не найдены {u_key} или {p_key} в окружении (.env)")
    return username, password


STANDS: Dict[str, StandConfig] = {
    "cloud": StandConfig(
        name="cloud",
        host="https://tagme.sberdevices.ru",
        cfg_path=_expand("~/configs/.ml_config.cfg"),
        seen_file="seen_tasks_cloud.json",
        organizations=[
            "e2f6ea8c-516a-4a43-aaa1-0544a9a60ec9",
            "2bdf1e09-59e1-469c-b055-f81891aa3e64",
            "d8212dcd-5025-4115-b5cd-531798cfba3c",
            "4f656097-5a22-43f2-b876-3cb568106b3d",
            "7c69d122-e8fd-4b34-950f-e0966935b3a5",
            "a2ca50d8-1a76-4323-8775-03e4dd2be0db",
            "7d2960ea-7f4e-48b8-824f-a432d7c77dc5",
        ],
        skip_project_names=[],
    ),
    "sigma": StandConfig(
        name="sigma",
        host="https://tagme.sigma.sbrf.ru",
        cfg_path=_expand("~/configs/.ml_config_sigma.cfg"),
        seen_file="seen_tasks_sigma.json",
        organizations=[
             "ce07feb6-7405-4d32-8346-efe1eb7b08df",
             "9d996393-ede9-47ac-beb7-8b0e232547f6",
             "42b12cba-87b2-459c-ae40-d0ca064f33aa",
             "2673e31b-8602-48c9-bf5f-8be78bfc6f47",
        ],
        skip_project_names=[],
        require_service=True,
        forbidden_services=frozenset({"vsellm"}),
    ),
    "sberdev": StandConfig(
        name="sberdev",
        host="https://tagme.sberdevices.ru",
        cfg_path=_expand("~/configs/.cloud_sberdevices.cfg"),
        seen_file="seen_tasks_sberdev.json",
        organizations=[
              "a2ca50d8-1a76-4323-8775-03e4dd2be0db",
              "e2f6ea8c-516a-4a43-aaa1-0544a9a60ec9",
              "1bf17a92-158c-4ff6-8e16-83a51386ab09",
              "2bdf1e09-59e1-469c-b055-f81891aa3e64",
              "d8212dcd-5025-4115-b5cd-531798cfba3c",
              "7d2960ea-7f4e-48b8-824f-a432d7c77dc5",
        ],
        skip_project_names=[
            "check dynamic overlap",
            "check lera 4",
            "Kandinsky__image_foto_celebrities_validate check",
            "Kandinsky_описание видео_АБТ_validate check",
            "LAP++____SBOL___MAIN___V7 check",
            "Malvina. Styles. RT_check",
            "MALVINA_Человечность промптов Malvina_штат check",
            "Pskov60_time_e2e + лист ос check",
            "shlenkin check OS LISTS",
            "test bundle check",
            "Автосохранение check",
            "Демонстрация бага check",
            "отложенная приемка check",
            "тест пайплайна сегментации",
            "Шаблон с большим количеством изображений check",
            "Разметка заявлений (Амурская область)"
        ],
    ),
}


def get_stand(name: str) -> StandConfig:
    key = (name or "").strip().lower()
    if key not in STANDS:
        raise KeyError(f"Unknown stand '{name}'. Available: {list(STANDS.keys())}")
    return STANDS[key]