"""
Модуль для загрузки отчётов на FIR (сетевое хранилище) через SMB протокол.

Использует библиотеку smbclient для загрузки Excel отчётов в указанную папку на FIR.
Учётные данные берутся из переменных окружения (.env файл).
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

import smbclient
from dotenv import load_dotenv

log = logging.getLogger(__name__)

# Загружаем .env файл
load_dotenv()


class FirUploader:
    """
    Класс для загрузки файлов на FIR через SMB протокол.

    Использует учётные данные из переменных окружения:
    - SMB_USERNAME: имя пользователя
    - SMB_PASSWORD: пароль
    - SMB_PORT: порт (по умолчанию 445)
    """

    def __init__(self, server: str = "bordo105.sigma.sbrf.ru", port: Optional[int] = None):
        """
        Инициализация загрузчика.

        Args:
            server: Адрес SMB сервера
            port: Порт (если None, берётся из env или 445)
        """
        self.server = server
        self.username = os.getenv("SMB_USERNAME")
        self.password = os.getenv("SMB_PASSWORD")
        self.port = port or int(os.getenv("SMB_PORT", "445"))

        if not self.username or not self.password:
            raise RuntimeError(
                "Не заданы SMB_USERNAME / SMB_PASSWORD в окружении (.env). "
                "Создайте файл .env с учётными данными."
            )

        # Регистрируем SMB сессию
        try:
            smbclient.register_session(
                self.server,
                username=self.username,
                password=self.password,
                port=self.port,
            )
            log.info(f"SMB сессия зарегистрирована | server={self.server} | port={self.port}")
        except Exception as e:
            log.error(f"Ошибка регистрации SMB сессии: {e}")
            raise

    def ensure_directory(self, remote_path: str) -> None:
        """
        Создаёт директорию на SMB, если её нет.

        Args:
            remote_path: Путь к директории на SMB (например, \\\\server\\share\\folder)
        """
        import smbclient.path

        try:
            if not smbclient.path.exists(remote_path):
                smbclient.makedirs(remote_path, exist_ok=True)
                log.info(f"Создана директория на FIR: {remote_path}")
            else:
                log.debug(f"Директория уже существует: {remote_path}")
        except Exception as e:
            log.error(f"Ошибка создания директории {remote_path}: {e}")
            raise

    def upload_file(self, local_path: Path, remote_path: str) -> bool:
        """
        Загружает файл на FIR.

        Args:
            local_path: Путь к локальному файлу
            remote_path: Полный путь к файлу на SMB (включая имя файла)

        Returns:
            True если загрузка успешна, False иначе
        """
        if not local_path.exists():
            log.error(f"Локальный файл не найден: {local_path}")
            return False

        try:
            # Создаём родительскую директорию, если нужно
            parent_dir = str(Path(remote_path).parent).replace("/", "\\")
            if not parent_dir.startswith("\\\\"):
                # Добавляем префикс UNC path если нужно
                parent_dir = f"\\\\{self.server}\\{parent_dir}"

            self.ensure_directory(parent_dir)

            # Загружаем файл
            log.info(f"Начинаем загрузку: {local_path.name} -> {remote_path}")

            with open(local_path, "rb") as local_f:
                with smbclient.open_file(remote_path, mode="wb") as smb_f:
                    shutil.copyfileobj(local_f, smb_f)

            log.info(f"✓ Файл успешно загружен на FIR: {remote_path}")
            return True

        except Exception as e:
            log.error(f"✗ Ошибка загрузки файла {local_path} на FIR: {e}", exc_info=True)
            return False

    def upload_file_to_project_folder(
        self,
        local_path: Path,
        base_fir_path: str,
        subfolder: Optional[str] = None
    ) -> bool:
        """
        Загружает файл в папку проекта на FIR.

        Args:
            local_path: Путь к локальному файлу
            base_fir_path: Базовый путь к папке проекта на FIR
            subfolder: Опциональная подпапка (например, дата "2026-05-31")

        Returns:
            True если загрузка успешна, False иначе
        """
        # Формируем полный путь на FIR
        if subfolder:
            remote_dir = f"{base_fir_path}\\{subfolder}"
        else:
            remote_dir = base_fir_path

        remote_file_path = f"{remote_dir}\\{local_path.name}"

        return self.upload_file(local_path, remote_file_path)


def upload_report_to_fir(
    report_path: Path,
    fir_base_path: str,
    run_date: Optional[str] = None
) -> bool:
    """
    Загружает отчёт на FIR.

    Args:
        report_path: Путь к локальному Excel отчёту
        fir_base_path: Базовый путь к папке на FIR (из конфига проекта)
        run_date: Дата запуска (для создания подпапки), например "2026-05-31"

    Returns:
        True если загрузка успешна, False иначе
    """
    try:
        uploader = FirUploader()
        return uploader.upload_file_to_project_folder(
            local_path=report_path,
            base_fir_path=fir_base_path,
            subfolder=run_date  # Создаём подпапку по дате
        )
    except Exception as e:
        log.error(f"Не удалось загрузить отчёт на FIR: {e}", exc_info=True)
        return False
