"""
Модуль для расчета метрик качества разметки
"""
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
import logging

log = logging.getLogger("quality_metrics")


def calculate_agreement_rate(annotations: List[Any]) -> float:
    """
    Вычисляет процент согласия между разметчиками.

    Args:
        annotations: Список аннотаций от разных разметчиков

    Returns:
        Процент согласия (0.0 - 1.0)
    """
    if not annotations or len(annotations) < 2:
        return 1.0

    # Подсчитываем самый частый ответ
    counter = Counter(str(ann) for ann in annotations)
    most_common_count = counter.most_common(1)[0][1]

    return most_common_count / len(annotations)


def calculate_overlap_agreement(assignments: List[Dict[str, Any]],
                                output_fields: List[str]) -> Dict[str, Any]:
    """
    Вычисляет согласие между разметчиками по перекрытию.

    Args:
        assignments: Список assignments для одного задания
        output_fields: Список полей с ответами для анализа

    Returns:
        Словарь с метриками согласия по каждому полю
    """
    if len(assignments) < 2:
        return {
            "overlap": len(assignments),
            "agreement_possible": False,
            "fields": {}
        }

    metrics = {
        "overlap": len(assignments),
        "agreement_possible": True,
        "fields": {}
    }

    for field in output_fields:
        values = []
        for assignment in assignments:
            output_data = assignment.get("output_data", {})
            if field in output_data:
                values.append(str(output_data[field]))

        if values:
            agreement = calculate_agreement_rate(values)
            unique_values = len(set(values))

            metrics["fields"][field] = {
                "agreement_rate": agreement,
                "unique_values": unique_values,
                "total_annotations": len(values),
                "values": values
            }

    # Общее согласие по всем полям
    if metrics["fields"]:
        avg_agreement = sum(
            f["agreement_rate"] for f in metrics["fields"].values()
        ) / len(metrics["fields"])
        metrics["average_agreement"] = avg_agreement
    else:
        metrics["average_agreement"] = 0.0

    return metrics


def calculate_cohens_kappa(annotations1: List[str],
                           annotations2: List[str]) -> Optional[float]:
    """
    Вычисляет Cohen's Kappa для двух разметчиков.

    Args:
        annotations1: Аннотации первого разметчика
        annotations2: Аннотации второго разметчика

    Returns:
        Cohen's Kappa коэффициент или None, если расчет невозможен
    """
    if len(annotations1) != len(annotations2) or len(annotations1) == 0:
        return None

    # Observed agreement
    agreements = sum(1 for a1, a2 in zip(annotations1, annotations2) if a1 == a2)
    po = agreements / len(annotations1)

    # Expected agreement
    categories = set(annotations1 + annotations2)
    pe = 0.0

    for category in categories:
        p1 = annotations1.count(category) / len(annotations1)
        p2 = annotations2.count(category) / len(annotations2)
        pe += p1 * p2

    # Cohen's Kappa
    if pe == 1.0:
        return 1.0

    kappa = (po - pe) / (1 - pe)
    return kappa


def identify_problem_assignments(assignments_by_task: Dict[str, List[Dict[str, Any]]],
                                 output_fields: List[str],
                                 agreement_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Определяет проблемные задания с низким согласием.

    Args:
        assignments_by_task: Словарь {task_id: [assignments]}
        output_fields: Список полей для анализа
        agreement_threshold: Порог согласия (задания ниже - проблемные)

    Returns:
        Список проблемных заданий с метриками
    """
    problem_tasks = []

    for task_id, assignments in assignments_by_task.items():
        if len(assignments) < 2:
            continue

        metrics = calculate_overlap_agreement(assignments, output_fields)

        if metrics.get("average_agreement", 1.0) < agreement_threshold:
            problem_tasks.append({
                "task_id": task_id,
                "overlap": len(assignments),
                "average_agreement": metrics.get("average_agreement", 0.0),
                "field_metrics": metrics.get("fields", {}),
                "assignments": assignments
            })

    # Сортируем по уровню несогласия (худшие - первые)
    problem_tasks.sort(key=lambda x: x["average_agreement"])

    return problem_tasks


def calculate_marker_quality(assignments: List[Dict[str, Any]],
                            output_fields: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Вычисляет метрики качества для каждого разметчика.

    Args:
        assignments: Список всех assignments
        output_fields: Список полей для анализа

    Returns:
        Словарь {marker_id: {metrics}}
    """
    marker_stats = defaultdict(lambda: {
        "total_assignments": 0,
        "agreements": 0,
        "disagreements": 0,
        "assignments_list": []
    })

    # Группируем по заданиям
    assignments_by_task = defaultdict(list)
    for assignment in assignments:
        task_id = assignment.get("input_data", {}).get("task_id", "unknown")
        assignments_by_task[task_id].append(assignment)

    # Анализируем каждое задание
    for task_id, task_assignments in assignments_by_task.items():
        if len(task_assignments) < 2:
            continue

        # Определяем мажоритарный ответ для каждого поля
        majority_answers = {}
        for field in output_fields:
            values = [
                str(a.get("output_data", {}).get(field, ""))
                for a in task_assignments
                if field in a.get("output_data", {})
            ]
            if values:
                counter = Counter(values)
                majority_answers[field] = counter.most_common(1)[0][0]

        # Сравниваем каждого разметчика с мажоритарным ответом
        for assignment in task_assignments:
            marker_id = assignment.get("marker_id", "unknown")
            marker_stats[marker_id]["total_assignments"] += 1
            marker_stats[marker_id]["assignments_list"].append(assignment)

            output_data = assignment.get("output_data", {})
            agrees = 0
            total_fields = 0

            for field in output_fields:
                if field in output_data and field in majority_answers:
                    total_fields += 1
                    if str(output_data[field]) == majority_answers[field]:
                        agrees += 1

            if total_fields > 0:
                if agrees == total_fields:
                    marker_stats[marker_id]["agreements"] += 1
                else:
                    marker_stats[marker_id]["disagreements"] += 1

    # Вычисляем финальные метрики
    result = {}
    for marker_id, stats in marker_stats.items():
        total = stats["agreements"] + stats["disagreements"]
        accuracy = stats["agreements"] / total if total > 0 else 0.0

        result[marker_id] = {
            "total_assignments": stats["total_assignments"],
            "agreements": stats["agreements"],
            "disagreements": stats["disagreements"],
            "accuracy": accuracy
        }

    return result


def calculate_task_quality_summary(assignments: List[Dict[str, Any]],
                                  output_fields: List[str]) -> Dict[str, Any]:
    """
    Вычисляет общую сводку по качеству задачи.

    Args:
        assignments: Список всех assignments
        output_fields: Список полей для анализа

    Returns:
        Словарь с метриками качества
    """
    if not assignments:
        return {
            "total_assignments": 0,
            "unique_tasks": 0,
            "average_overlap": 0.0,
            "average_agreement": 0.0,
            "problem_tasks_count": 0
        }

    # Группируем по заданиям
    assignments_by_task = defaultdict(list)
    for assignment in assignments:
        task_id = assignment.get("input_data", {}).get("task_id",
                                assignment.get("assignment_id", "unknown"))
        assignments_by_task[task_id].append(assignment)

    # Вычисляем метрики
    overlaps = [len(assigns) for assigns in assignments_by_task.values()]
    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0

    agreements = []
    for task_id, task_assignments in assignments_by_task.items():
        if len(task_assignments) >= 2:
            metrics = calculate_overlap_agreement(task_assignments, output_fields)
            agreements.append(metrics.get("average_agreement", 0.0))

    avg_agreement = sum(agreements) / len(agreements) if agreements else 1.0

    # Подсчет проблемных заданий
    problem_tasks = identify_problem_assignments(assignments_by_task, output_fields)

    return {
        "total_assignments": len(assignments),
        "unique_tasks": len(assignments_by_task),
        "average_overlap": avg_overlap,
        "average_agreement": avg_agreement,
        "problem_tasks_count": len(problem_tasks),
        "problem_tasks_rate": len(problem_tasks) / len(assignments_by_task) if assignments_by_task else 0.0
    }
