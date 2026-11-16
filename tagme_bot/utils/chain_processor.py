#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chain_processor.py

Универсальный обработчик цепочек промптов.

Инструкция должна содержать блоки:

    <PROMPT_1>
    Первый промпт.
    Вход пользователя: {{user_input}}
    </PROMPT_1>

    <PROMPT_2>
    Используй ответ предыдущего шага:
    {{answer_1}}
    </PROMPT_2>

Поддерживаемые переменные:

    {{user_input}}  — исходный запрос пользователя;
    {{answer_1}}    — ответ первого шага;
    {{answer_2}}    — ответ второго шага;
    {{think_1}}     — рассуждение первого шага, если модель
                      вернула блок <think>...</think>.

Вызов модели передаётся через параметр call_model. Благодаря этому
в цепочке можно использовать тот же универсальный call_vsellm,
который уже поддерживает VseLLM, direct, GigaHF, IFT и другие роуты.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


ModelCallResult = Union[str, Dict[str, Any]]
ModelCallable = Callable[..., ModelCallResult]


class ChainProcessor:
    """
    Последовательно выполняет блоки <PROMPT_N> через переданную
    функцию call_model.

    call_model вызывается так:

        call_model(
            prompt=prompt,
            model=model,
            service=service,
            system_prompt=system_prompt,
            cfg=cfg,
            step=step,
            **model_call_kwargs,
        )

    В vsellm client.py мы затем добавим небольшой адаптер,
    который преобразует эти параметры к фактической сигнатуре
    call_vsellm().
    """

    PROMPT_PATTERN = re.compile(
        r"<PROMPT_(\d+)>\s*([\s\S]*?)\s*</PROMPT_\1>",
        flags=re.IGNORECASE,
    )

    PLACEHOLDER_PATTERN = re.compile(
        r"\{\{\s*([^{}]+?)\s*\}\}"
    )

    THINK_PATTERN = re.compile(
        r"<think\b[^>]*>([\s\S]*?)(?:</think>|$)",
        flags=re.IGNORECASE,
    )

    DEFAULT_SYSTEM_PROMPT = "Следуй инструкциям пользователя."

    def __init__(
        self,
        instr: str,
        model: str,
        cfg: Optional[Dict[str, Any]],
        *,
        call_model: ModelCallable,
        service: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model_call_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            instr:
                Полная инструкция с блоками
                <PROMPT_1>...</PROMPT_1>.

            model:
                Название модели, которое будет передаваться
                в универсальный клиент.

            cfg:
                Параметры генерации, например:

                    {
                        "temperature": 0,
                        "max_tokens": 2000,
                        "top_p": 1,
                        "think": False
                    }

            call_model:
                Функция, через которую выполняется один шаг цепочки.

            service:
                Роут модели, например:
                vsellm, direct, gigahf, ift.

            system_prompt:
                Системный промпт для каждого вызова модели.

            model_call_kwargs:
                Дополнительные параметры, которые нужно передавать
                в функцию call_model на каждом шаге.
        """

        if not isinstance(instr, str):
            raise TypeError(
                "instr должен быть строкой, "
                f"получено: {type(instr).__name__}"
            )

        if not instr.strip():
            raise ValueError("instr не должен быть пустым")

        if not isinstance(model, str) or not model.strip():
            raise ValueError("model должен быть непустой строкой")

        if not callable(call_model):
            raise TypeError("call_model должен быть вызываемой функцией")

        self.instr = instr
        self.model = model.strip()
        self.service = service

        self.cfg: Dict[str, Any] = dict(cfg or {})
        self.call_model = call_model

        self.system_prompt = (
            system_prompt
            or self.cfg.get("system_prompt")
            or self.DEFAULT_SYSTEM_PROMPT
        )

        self.model_call_kwargs: Dict[str, Any] = dict(
            model_call_kwargs or {}
        )

        self.blocks = self._parse_blocks(instr)

        self.prompts: Dict[int, str] = {}
        self.step_usage: Dict[int, Dict[str, int]] = {}
        self.total_usage: Dict[str, int] = self._empty_usage()

    # ------------------------------------------------------------------
    # Парсинг цепочки
    # ------------------------------------------------------------------

    @classmethod
    def _parse_blocks(
        cls,
        instr: str,
    ) -> List[Tuple[int, str]]:
        """
        Извлекает все блоки <PROMPT_N> и сортирует их по номеру.

        Returns:
            Список:

                [
                    (1, "текст первого промпта"),
                    (2, "текст второго промпта")
                ]
        """

        blocks = [
            (
                int(match.group(1)),
                match.group(2).strip(),
            )
            for match in cls.PROMPT_PATTERN.finditer(instr)
        ]

        if not blocks:
            raise ValueError(
                "В инструкции не найдены блоки вида "
                "<PROMPT_1>...</PROMPT_1>"
            )

        blocks.sort(key=lambda item: item[0])

        seen_numbers = set()

        for block_number, block_text in blocks:
            if block_number in seen_numbers:
                raise ValueError(
                    f"В инструкции повторяется PROMPT_{block_number}"
                )

            if not block_text:
                raise ValueError(
                    f"Блок PROMPT_{block_number} пустой"
                )

            seen_numbers.add(block_number)

        return blocks

    # ------------------------------------------------------------------
    # Подстановка переменных
    # ------------------------------------------------------------------

    @classmethod
    def _replace_placeholders(
        cls,
        text: str,
        values: Dict[str, Any],
    ) -> str:
        """
        Подставляет значения в placeholders:

            {{user_input}}
            {{answer_1}}
            {{answer_2}}
            {{think_1}}

        Неизвестные placeholders остаются в тексте без изменений.
        """

        def replace(match: re.Match[str]) -> str:
            key = match.group(1).strip()

            if key not in values:
                return match.group(0)

            value = values[key]

            if value is None:
                return ""

            if isinstance(value, str):
                return value

            if isinstance(value, (dict, list)):
                return json.dumps(
                    value,
                    ensure_ascii=False,
                )

            return str(value)

        return cls.PLACEHOLDER_PATTERN.sub(
            replace,
            text,
        )

    # ------------------------------------------------------------------
    # Работа с think
    # ------------------------------------------------------------------

    @classmethod
    def _split_think(
        cls,
        raw_answer: Any,
    ) -> Tuple[str, str]:
        """
        Разделяет ответ модели на reasoning и итоговый ответ.

        Например:

            <think>
            Внутреннее рассуждение
            </think>

            Финальный ответ

        Returns:
            (
                "Внутреннее рассуждение",
                "Финальный ответ"
            )
        """

        if not isinstance(raw_answer, str):
            return "", str(raw_answer)

        think_parts = [
            match.group(1).strip()
            for match in cls.THINK_PATTERN.finditer(raw_answer)
            if match.group(1).strip()
        ]

        if not think_parts:
            return "", raw_answer.strip()

        think_text = "\n---\n".join(think_parts)

        clean_answer = cls.THINK_PATTERN.sub(
            "",
            raw_answer,
        ).strip()

        return think_text, clean_answer

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_usage() -> Dict[str, int]:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _normalize_usage(
        cls,
        usage: Any,
    ) -> Dict[str, int]:
        """
        Приводит разные форматы usage к одному виду.
        """

        if not isinstance(usage, dict):
            return cls._empty_usage()

        prompt_tokens = cls._safe_int(
            usage.get(
                "prompt_tokens",
                usage.get(
                    "input_tokens",
                    usage.get(
                        "promptTokenCount",
                        0,
                    ),
                ),
            )
        )

        completion_tokens = cls._safe_int(
            usage.get(
                "completion_tokens",
                usage.get(
                    "output_tokens",
                    usage.get(
                        "candidatesTokenCount",
                        0,
                    ),
                ),
            )
        )

        total_tokens = cls._safe_int(
            usage.get(
                "total_tokens",
                usage.get(
                    "totalTokenCount",
                    0,
                ),
            )
        )

        if total_tokens <= 0:
            total_tokens = (
                prompt_tokens
                + completion_tokens
            )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _add_usage(
        self,
        step: int,
        usage: Any,
    ) -> None:
        normalized_usage = self._normalize_usage(usage)

        self.step_usage[step] = normalized_usage

        for key in self.total_usage:
            self.total_usage[key] += normalized_usage[key]

    # ------------------------------------------------------------------
    # Парсинг ответа универсального клиента
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_usage(
        result: ModelCallResult,
    ) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return {}

        usage = result.get("usage")

        if isinstance(usage, dict):
            return usage

        raw = result.get("raw")

        if isinstance(raw, dict):
            nested_usage = raw.get("usage")

            if isinstance(nested_usage, dict):
                return nested_usage

        return {}

    @staticmethod
    def _extract_answer(
        result: ModelCallResult,
    ) -> str:
        """
        Извлекает текст ответа из разных форматов универсального клиента.
        """

        if isinstance(result, str):
            return result.strip()

        if not isinstance(result, dict):
            return str(result).strip()

        possible_keys = (
            "answer",
            "model_answer",
            "raw_text",
            "text",
            "content",
            "response",
            "result",
            "output",
        )

        for key in possible_keys:
            value = result.get(key)

            if isinstance(value, str) and value.strip():
                return value.strip()

            if isinstance(value, (dict, list)):
                return json.dumps(
                    value,
                    ensure_ascii=False,
                )

        raw = result.get("raw")

        if isinstance(raw, str) and raw.strip():
            return raw.strip()

        if isinstance(raw, (dict, list)):
            return json.dumps(
                raw,
                ensure_ascii=False,
            )

        return ""

    # ------------------------------------------------------------------
    # Вызов одного шага
    # ------------------------------------------------------------------

    def _call_once(
        self,
        prompt: str,
        step: int,
    ) -> str:
        """
        Выполняет один шаг через переданный универсальный клиент.
        """

        print(
            f"[CHAIN_CALL] "
            f"step={step} "
            f"service={self.service} "
            f"model={self.model} "
            f"prompt_len={len(prompt)}"
        )

        call_kwargs = dict(self.model_call_kwargs)

        call_kwargs.update(
            {
                "prompt": prompt,
                "model": self.model,
                "service": self.service,
                "system_prompt": self.system_prompt,
                "cfg": dict(self.cfg),
                "step": step,
            }
        )

        try:
            result = self.call_model(**call_kwargs)

        except Exception as error:
            print(
                f"[CHAIN_CALL_ERROR] "
                f"step={step} "
                f"service={self.service} "
                f"model={self.model} "
                f"error_type={type(error).__name__} "
                f"error={error}"
            )
            raise

        usage = self._extract_usage(result)
        self._add_usage(step, usage)

        answer = self._extract_answer(result)

        if not answer:
            raise RuntimeError(
                f"Модель вернула пустой ответ "
                f"на шаге {step}"
            )

        return answer

    # ------------------------------------------------------------------
    # Остановка цепочки
    # ------------------------------------------------------------------

    @staticmethod
    def _is_exit_answer(answer: str) -> bool:
        """
        Останавливает цепочку только если exit является отдельным словом.

        Сработает:

            exit
            EXIT
            Результат: exit

        Не сработает:

            exiting
            no_exit_value
        """

        return bool(
            re.search(
                r"(?<![\wа-яё])exit(?![\wа-яё])",
                answer,
                flags=re.IGNORECASE,
            )
        )

    # ------------------------------------------------------------------
    # Запуск цепочки
    # ------------------------------------------------------------------

    def _reset_runtime_state(self) -> None:
        self.prompts.clear()
        self.step_usage.clear()
        self.total_usage = self._empty_usage()

    def run_chain(
        self,
        user_input: Any,
    ) -> Dict[str, Any]:
        """
        Последовательно выполняет все PROMPT-блоки.

        Returns:
            Например:

                {
                    "answer_1": "...",
                    "answer_2": "...",
                    "think_1": "..."
                }
        """

        self._reset_runtime_state()

        normalized_user_input = (
            user_input
            if isinstance(user_input, str)
            else json.dumps(
                user_input,
                ensure_ascii=False,
            )
        )

        answers: Dict[str, Any] = {}

        print("=== CHAIN START ===")
        print(
            f"[CHAIN_INPUT] "
            f"len={len(normalized_user_input)}"
        )

        for sequence_step, (
            prompt_number,
            prompt_template,
        ) in enumerate(
            self.blocks,
            start=1,
        ):
            placeholder_values: Dict[str, Any] = {
                "user_input": normalized_user_input,
                **answers,
            }

            prepared_prompt = self._replace_placeholders(
                prompt_template,
                placeholder_values,
            )

            self.prompts[sequence_step] = prepared_prompt

            raw_answer = self._call_once(
                prompt=prepared_prompt,
                step=sequence_step,
            )

            think_text, clean_answer = self._split_think(
                raw_answer
            )

            if think_text:
                answers[
                    f"think_{sequence_step}"
                ] = think_text

            answers[
                f"answer_{sequence_step}"
            ] = clean_answer

            print(
                f"[CHAIN_ANSWER] "
                f"step={sequence_step} "
                f"prompt_number={prompt_number} "
                f"answer_len={len(clean_answer)}"
            )

            if self._is_exit_answer(clean_answer):
                print(
                    f"[CHAIN_EXIT] "
                    f"step={sequence_step}"
                )
                break

        print("=== CHAIN END ===")

        return answers

    # ------------------------------------------------------------------
    # Формирование итогового ответа
    # ------------------------------------------------------------------

    @staticmethod
    def _get_last_answer_key(
        chain_answers: Dict[str, Any],
    ) -> Optional[str]:
        answer_keys = [
            key
            for key in chain_answers
            if re.fullmatch(
                r"answer_\d+",
                key,
            )
        ]

        if not answer_keys:
            return None

        answer_keys.sort(
            key=lambda key: int(
                key.rsplit("_", 1)[-1]
            )
        )

        return answer_keys[-1]

    @staticmethod
    def _clean_json_text(
        raw_text: str,
    ) -> str:
        """
        Убирает markdown-обёртку вокруг JSON.
        """

        cleaned = raw_text.strip()

        cleaned = re.sub(
            r"^\s*```json\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        cleaned = re.sub(
            r"^\s*```\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```\s*$",
            "",
            cleaned,
        )

        return cleaned.strip()

    def run_and_build_payload(
        self,
        user_input: Any,
    ) -> Dict[str, Any]:
        """
        Запускает цепочку и формирует общий результат.

        Если последний ответ является JSON, его поля переносятся
        в корень payload.

        Если последний ответ не является JSON, он возвращается
        в model_answer.
        """

        chain_answers = self.run_chain(user_input)

        last_answer_key = self._get_last_answer_key(
            chain_answers
        )

        if last_answer_key is None:
            final_answer = ""
            completed_steps = 0

        else:
            final_answer = str(
                chain_answers.get(
                    last_answer_key,
                    "",
                )
            )

            completed_steps = int(
                last_answer_key.rsplit(
                    "_",
                    1,
                )[-1]
            )

        cleaned_final_answer = self._clean_json_text(
            final_answer
        )

        try:
            parsed_answer = json.loads(
                cleaned_final_answer
            )

            if isinstance(parsed_answer, dict):
                payload: Dict[str, Any] = parsed_answer

            else:
                payload = {
                    "result": parsed_answer,
                }

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            payload = {
                "model_answer": final_answer,
            }

        payload["answer"] = final_answer
        payload["raw_text"] = final_answer

        payload["chain_answers"] = chain_answers
        payload["prompts"] = dict(self.prompts)

        payload["usage"] = dict(self.total_usage)
        payload["step_usage"] = dict(
            self.step_usage
        )

        payload["chain_meta"] = {
            "enabled": True,
            "model": self.model,
            "service": self.service,
            "total_steps": len(self.blocks),
            "completed_steps": completed_steps,
            "stopped_early": (
                completed_steps < len(self.blocks)
            ),
        }

        return payload