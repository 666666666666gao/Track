#!/usr/bin/env python3
import argparse
import gzip
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


GATES = {
    "candidate_probability": 0.50,
    "candidate_margin": 0.10,
    "beneficial_probability": 0.80,
    "catastrophic_probability_max": 0.05,
    "predicted_gain": 0.05,
}


def read_jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantiles(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {"min": None, "q25": None, "median": None, "q75": None, "max": None, "mean": None}

    def q(p):
        if len(values) == 1:
            return values[0]
        location = (len(values) - 1) * p
        low = int(math.floor(location))
        high = int(math.ceil(location))
        if low == high:
            return values[low]
        fraction = location - low
        return values[low] * (1.0 - fraction) + values[high] * fraction

    return {
        "min": values[0],
        "q25": q(0.25),
        "median": q(0.50),
        "q75": q(0.75),
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def binary_auc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        rank_sum += average_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels, scores):
    positives = sum(labels)
    if positives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(ordered, start=1):
        if label:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return None
    return numerator / (dx * dy)


def rankdata(values):
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        for position in range(index, end):
            ranks[ordered[position][0]] = average_rank
        index = end
    return ranks


def event_summary(events):
    for event in events:
        if "event_class" not in event:
            if any(bool(action["actual_beneficial"]) for action in event["actions"]):
                event["event_class"] = "beneficial"
            elif any(bool(action["actual_catastrophic"]) for action in event["actions"]):
                event["event_class"] = "catastrophic"
            else:
                event["event_class"] = "neutral"
    all_actions = [action for event in events for action in event["actions"]]
    result = {
        "events": len(events),
        "sequences": len({event["sequence"] for event in events}),
        "event_classes": dict(Counter(event["event_class"] for event in events)),
        "actions": len(all_actions),
        "action_labels": {
            "beneficial": sum(bool(action["actual_beneficial"]) for action in all_actions),
            "catastrophic": sum(bool(action["actual_catastrophic"]) for action in all_actions),
            "neutral": sum(not action["actual_beneficial"] and not action["actual_catastrophic"] for action in all_actions),
        },
    }

    top_rows = []
    gate_counts = Counter()
    sequential_counts = Counter()
    class_gate_counts = defaultdict(Counter)
    label_contract_cross = Counter()
    beneficial_event_ranks = []
    beneficial_event_top_correct = 0
    beneficial_event_candidate_beats_abstain = 0
    duplicate_current_last = Counter()
    unique_signatures = []
    sequence_rows = defaultdict(lambda: Counter(events=0, beneficial_events=0, catastrophic_events=0, neutral_events=0))

    for event in events:
        actions = event["actions"]
        top = max(actions, key=lambda action: action["selection_probability"])
        top_selection = float(top["selection_probability"])
        abstain = float(event["abstain_probability"])
        conditions = {
            "beats_abstain": top_selection > abstain,
            "candidate_probability": top_selection >= GATES["candidate_probability"],
            "candidate_margin": top_selection - abstain >= GATES["candidate_margin"],
            "beneficial_probability": float(top["beneficial_probability"]) >= GATES["beneficial_probability"],
            "catastrophic_probability": float(top["catastrophic_probability"]) <= GATES["catastrophic_probability_max"],
            "predicted_gain": float(top["predicted_gain"]) >= GATES["predicted_gain"],
        }
        for name, passed in conditions.items():
            gate_counts[name] += int(passed)
            class_gate_counts[event["event_class"]][name] += int(passed)
        ordered_gates = [
            "beats_abstain",
            "candidate_probability",
            "candidate_margin",
            "beneficial_probability",
            "catastrophic_probability",
            "predicted_gain",
        ]
        prefix = True
        for name in ordered_gates:
            prefix = prefix and conditions[name]
            sequential_counts[name] += int(prefix)
        gate_counts["all"] += int(all(conditions.values()))
        class_gate_counts[event["event_class"]]["all"] += int(all(conditions.values()))
        loose_class = (
            "beneficial" if any(bool(action["actual_beneficial"]) for action in actions)
            else "catastrophic" if any(bool(action["actual_catastrophic"]) for action in actions)
            else "neutral"
        )
        label_contract_cross[(event["event_class"], loose_class)] += 1
        top_rows.append(
            {
                "event_class": event["event_class"],
                "selection": top_selection,
                "abstain": abstain,
                "margin": top_selection - abstain,
                "benefit": float(top["beneficial_probability"]),
                "catastrophe": float(top["catastrophic_probability"]),
                "predicted_gain": float(top["predicted_gain"]),
                "actual_gain": float(top["actual_gain"]),
                "actual_beneficial": bool(top["actual_beneficial"]),
                "actual_catastrophic": bool(top["actual_catastrophic"]),
                "name": top["name"],
            }
        )
        seq = sequence_rows[event["sequence"]]
        seq["events"] += 1
        seq[f"{event['event_class']}_events"] += 1
        seq["top_actual_beneficial"] += int(top["actual_beneficial"])
        seq["top_actual_catastrophic"] += int(top["actual_catastrophic"])

        if event["event_class"] == "beneficial":
            ranked = sorted(actions, key=lambda action: action["selection_probability"], reverse=True)
            best_rank = min(index + 1 for index, action in enumerate(ranked) if action["actual_beneficial"])
            beneficial_event_ranks.append(best_rank)
            beneficial_event_top_correct += int(top["actual_beneficial"])
            beneficial_event_candidate_beats_abstain += int(
                max(action["selection_probability"] for action in actions if action["actual_beneficial"]) > abstain
            )

        by_name = {action["name"]: action for action in actions}
        for peak in (0, 1):
            left = by_name[f"current_peak{peak}"]
            right = by_name[f"last_reliable_peak{peak}"]
            fields = (
                "selection_probability",
                "beneficial_probability",
                "catastrophic_probability",
                "predicted_gain",
                "refined_response",
                "ious",
            )
            duplicate_current_last[f"peak{peak}_identical"] += int(all(left[field] == right[field] for field in fields))

        signatures = {
            (
                round(float(action["selection_probability"]), 9),
                round(float(action["beneficial_probability"]), 9),
                round(float(action["catastrophic_probability"]), 9),
                round(float(action["predicted_gain"]), 9),
                tuple(round(float(value), 9) for value in action["ious"]),
            )
            for action in actions
        }
        unique_signatures.append(len(signatures))

    result["independent_gate_pass_counts"] = dict(gate_counts)
    result["sequential_gate_pass_counts"] = dict(sequential_counts)
    result["gate_pass_counts_by_strict_event_class"] = {
        event_class: dict(counts)
        for event_class, counts in sorted(class_gate_counts.items())
    }
    result["strict_h10_vs_loose_h4_event_class"] = {
        f"strict_{strict}__h4_{loose}": count
        for (strict, loose), count in sorted(label_contract_cross.items())
    }
    result["strict_h10_vs_loose_h4_mismatch_events"] = sum(
        count for (strict, loose), count in label_contract_cross.items()
        if strict != loose
    )
    result["top_candidate_distributions"] = {
        field: quantiles(row[field] for row in top_rows)
        for field in ("selection", "abstain", "margin", "benefit", "catastrophe", "predicted_gain", "actual_gain")
    }
    result["top_candidate_actual"] = {
        "beneficial": sum(row["actual_beneficial"] for row in top_rows),
        "catastrophic": sum(row["actual_catastrophic"] for row in top_rows),
        "neutral": sum(not row["actual_beneficial"] and not row["actual_catastrophic"] for row in top_rows),
    }
    result["beneficial_event_ranking"] = {
        "events": len(beneficial_event_ranks),
        "best_beneficial_is_top_candidate": beneficial_event_top_correct,
        "best_beneficial_beats_abstain": beneficial_event_candidate_beats_abstain,
        "best_beneficial_rank_distribution_among_six": quantiles(beneficial_event_ranks),
    }
    result["duplicate_branch_outputs"] = {
        **dict(duplicate_current_last),
        "events": len(events),
        "unique_action_signatures_per_event": quantiles(unique_signatures),
    }

    result["action_discrimination"] = {}
    for score_name, label_name in (
        ("selection_probability", "actual_beneficial"),
        ("beneficial_probability", "actual_beneficial"),
        ("predicted_gain", "actual_beneficial"),
        ("catastrophic_probability", "actual_catastrophic"),
    ):
        labels = [int(action[label_name]) for action in all_actions]
        scores = [float(action[score_name]) for action in all_actions]
        result["action_discrimination"][f"{score_name}_for_{label_name}"] = {
            "auc": binary_auc(labels, scores),
            "average_precision": average_precision(labels, scores),
            "positives": sum(labels),
            "total": len(labels),
        }

    result["event_discrimination_for_strict_h10_class"] = {}
    event_score_fields = {
        "one_minus_abstain": [1.0 - row["abstain"] for row in top_rows],
        "top_selection": [row["selection"] for row in top_rows],
        "top_benefit": [row["benefit"] for row in top_rows],
        "top_catastrophe": [row["catastrophe"] for row in top_rows],
        "top_predicted_gain": [row["predicted_gain"] for row in top_rows],
    }
    for target_class in ("beneficial", "catastrophic"):
        labels = [int(row["event_class"] == target_class) for row in top_rows]
        result["event_discrimination_for_strict_h10_class"][target_class] = {
            name: {
                "auc": binary_auc(labels, scores),
                "average_precision": average_precision(labels, scores),
            }
            for name, scores in event_score_fields.items()
        }

    predicted_gain = [float(action["predicted_gain"]) for action in all_actions]
    actual_gain = [float(action["actual_gain"]) for action in all_actions]
    result["gain_regression"] = {
        "pearson": pearson(predicted_gain, actual_gain),
        "spearman": pearson(rankdata(predicted_gain), rankdata(actual_gain)),
        "predicted": quantiles(predicted_gain),
        "actual": quantiles(actual_gain),
    }

    result["class_conditioned_top"] = {}
    for event_class in sorted({row["event_class"] for row in top_rows}):
        rows = [row for row in top_rows if row["event_class"] == event_class]
        result["class_conditioned_top"][event_class] = {
            "events": len(rows),
            "top_actual_beneficial": sum(row["actual_beneficial"] for row in rows),
            "top_actual_catastrophic": sum(row["actual_catastrophic"] for row in rows),
            **{
                field: quantiles(row[field] for row in rows)
                for field in ("selection", "abstain", "margin", "benefit", "catastrophe", "predicted_gain", "actual_gain")
            },
        }

    result["sequence_table"] = [
        {"sequence": name, **dict(counts)}
        for name, counts in sorted(sequence_rows.items(), key=lambda item: (-item[1]["beneficial_events"], item[0]))
    ]
    return result


def trace_summary(rows):
    has_optimizer_step = all("optimizer_step" in row for row in rows)
    has_event_class = all("event_class" in row for row in rows)
    output = {
        "rows": len(rows),
        "epochs": sorted({int(row["epoch"]) for row in rows}),
        "optimizer_steps": len({int(row["optimizer_step"]) for row in rows}) if has_optimizer_step else len(rows),
        "nonzero_gradient_rows": sum(float(row["gradient_norm"]) > 0 for row in rows),
        "by_epoch_class": {},
    }
    fields = [
        "loss",
        "setwise_total",
        "setwise_selection",
        "setwise_beneficial",
        "setwise_catastrophic",
        "setwise_gain",
        "setwise_pairwise",
        "setwise_beneficial_gate",
        "setwise_catastrophic_gate",
        "setwise_gain_gate",
        "dense_total",
    ]
    for epoch in output["epochs"]:
        event_classes = ("beneficial", "catastrophic", "neutral") if has_event_class else ("all",)
        for event_class in event_classes:
            subset = [
                row for row in rows
                if int(row["epoch"]) == epoch and (event_class == "all" or row["event_class"] == event_class)
            ]
            output["by_epoch_class"][f"epoch{epoch}_{event_class}"] = {
                "rows": len(subset),
                **{field: quantiles(float(row[field]) for row in subset) for field in fields},
            }
    return output


def fmt(value, digits=6):
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def render_markdown(report):
    m7 = report["m7"]["evaluation"]
    m6 = report["m6"]["evaluation"]
    lines = [
        "# STTrack LACHTT M7 full-set balanced 只读结果审计",
        "",
        "## 原始对照表",
        "",
        "|项目|M6 pilot|M7 full-set balanced|变化|",
        "|---|---:|---:|---:|",
        f"|held-out可用事件|{m6['events']}|{m7['events']}|{m7['events']-m6['events']:+d}|",
        f"|held-out序列|{m6['sequences']}|{m7['sequences']}|{m7['sequences']-m6['sequences']:+d}|",
        f"|selected actions|{m6['independent_gate_pass_counts']['all']}|{m7['independent_gate_pass_counts']['all']}|0|",
        f"|top candidate selection中位数|{fmt(m6['top_candidate_distributions']['selection']['median'])}|{fmt(m7['top_candidate_distributions']['selection']['median'])}|{fmt(m7['top_candidate_distributions']['selection']['median']-m6['top_candidate_distributions']['selection']['median'])}|",
        f"|abstain中位数|{fmt(m6['top_candidate_distributions']['abstain']['median'])}|{fmt(m7['top_candidate_distributions']['abstain']['median'])}|{fmt(m7['top_candidate_distributions']['abstain']['median']-m6['top_candidate_distributions']['abstain']['median'])}|",
        f"|benefit中位数|{fmt(m6['top_candidate_distributions']['benefit']['median'])}|{fmt(m7['top_candidate_distributions']['benefit']['median'])}|{fmt(m7['top_candidate_distributions']['benefit']['median']-m6['top_candidate_distributions']['benefit']['median'])}|",
        f"|catastrophe中位数|{fmt(m6['top_candidate_distributions']['catastrophe']['median'])}|{fmt(m7['top_candidate_distributions']['catastrophe']['median'])}|{fmt(m7['top_candidate_distributions']['catastrophe']['median']-m6['top_candidate_distributions']['catastrophe']['median'])}|",
        f"|predicted gain中位数|{fmt(m6['top_candidate_distributions']['predicted_gain']['median'])}|{fmt(m7['top_candidate_distributions']['predicted_gain']['median'])}|{fmt(m7['top_candidate_distributions']['predicted_gain']['median']-m6['top_candidate_distributions']['predicted_gain']['median'])}|",
        "",
        "## M7冻结门分解",
        "",
        "|条件|独立通过事件|按部署顺序累计通过|",
        "|---|---:|---:|",
    ]
    for key in ("beats_abstain", "candidate_probability", "candidate_margin", "beneficial_probability", "catastrophic_probability", "predicted_gain"):
        lines.append(
            f"|{key}|{m7['independent_gate_pass_counts'][key]}/{m7['events']}|{m7['sequential_gate_pass_counts'][key]}/{m7['events']}|"
        )
    lines.extend(
        [
            f"|全部条件|{m7['independent_gate_pass_counts']['all']}/{m7['events']}|{m7['independent_gate_pass_counts']['all']}/{m7['events']}|",
            "",
            "## M7候选区分能力",
            "",
            "|分数→真实标签|AUC|AP|正例/总数|",
            "|---|---:|---:|---:|",
        ]
    )
    for name, values in m7["action_discrimination"].items():
        lines.append(f"|{name}|{fmt(values['auc'])}|{fmt(values['average_precision'])}|{values['positives']}/{values['total']}|")
    rank = m7["beneficial_event_ranking"]
    duplicate = m7["duplicate_branch_outputs"]
    strict_benefit_event = m7["event_discrimination_for_strict_h10_class"]["beneficial"]
    class_gates = m7["gate_pass_counts_by_strict_event_class"]
    lines.extend(
        [
            "",
            "## 严格H10事件类与训练内H4动作标签契约",
            "",
            "M7均衡采样使用Gate-A严格H10事件类；setwise loss与最终`actual_*`使用较宽松H4动作定义。两者不是同一个标签。交叉计数如下：",
            "",
            "|严格H10事件类→宽松H4事件类|事件数|",
            "|---|---:|",
            *[
                f"|{name}|{count}|"
                for name, count in m7["strict_h10_vs_loose_h4_event_class"].items()
            ],
            f"|不一致合计|{m7['strict_h10_vs_loose_h4_mismatch_events']}|",
            "",
            "严格事件类上的关键门：",
            "",
            "|严格事件类|事件|candidate beats abstain|selection≥0.50|benefit≥0.80|全部门|",
            "|---|---:|---:|---:|---:|---:|",
            f"|beneficial|{m7['event_classes'].get('beneficial',0)}|{class_gates['beneficial'].get('beats_abstain',0)}|{class_gates['beneficial'].get('candidate_probability',0)}|{class_gates['beneficial'].get('beneficial_probability',0)}|{class_gates['beneficial'].get('all',0)}|",
            f"|catastrophic|{m7['event_classes'].get('catastrophic',0)}|{class_gates['catastrophic'].get('beats_abstain',0)}|{class_gates['catastrophic'].get('candidate_probability',0)}|{class_gates['catastrophic'].get('beneficial_probability',0)}|{class_gates['catastrophic'].get('all',0)}|",
            f"|neutral|{m7['event_classes'].get('neutral',0)}|{class_gates['neutral'].get('beats_abstain',0)}|{class_gates['neutral'].get('candidate_probability',0)}|{class_gates['neutral'].get('beneficial_probability',0)}|{class_gates['neutral'].get('all',0)}|",
            "",
            "## 关键发现",
            "",
            f"1. **观察：M7仍为0动作，但候选内排序已经出现正信号。** 154个可用held-out事件中完整门通过0；11个严格H10 beneficial事件里，宽松H4 beneficial候选成为六候选top-1的是{rank['best_beneficial_is_top_candidate']}/{rank['events']}。**解释：**当前最主要的失败已从“六个候选里找不到较好候选”转成“无法判断什么时候允许候选接管”。**含义：**不能进入low22，但不应把整个candidate representation简单判死。",
            f"2. **观察：abstain/commit路由方向错误。** 严格beneficial事件只有{class_gates['beneficial'].get('beats_abstain',0)}/{m7['event_classes'].get('beneficial',0)}个候选压过abstain，严格neutral却有{class_gates['neutral'].get('beats_abstain',0)}/{m7['event_classes'].get('neutral',0)}个；严格beneficial的`1-abstain`事件级AUC仅{fmt(strict_benefit_event['one_minus_abstain']['auc'])}。**解释：**均衡采样按严格H10标签分桶，但selection/benefit/gain的监督仍来自另一套宽松H4标签，commit语义没有被单独学习。**含义：**继续扫描现有阈值或epoch不会修复标签契约。",
            f"3. **观察：两套标签在{m7['strict_h10_vs_loose_h4_mismatch_events']}/{m7['events']}个held-out事件上不一致。** **解释：**这不是文件损坏，而是规范中同时存在Gate-A H10 `gain≥0.20/mean IoU≥0.50/early hits`与M7 H4 `gain≥0.05/no new low`两套定义；前者控制采样，后者控制loss和`actual_*`。**含义：**M7回答的是混合问题，无法直接学到VOT所需的长期survival commit。",
            f"4. **观察：动作级区分仍然偏弱。** selection→H4 beneficial AUC={fmt(m7['action_discrimination']['selection_probability_for_actual_beneficial']['auc'])}，benefit head AUC={fmt(m7['action_discrimination']['beneficial_probability_for_actual_beneficial']['auc'])}，cat head AUC={fmt(m7['action_discrimination']['catastrophic_probability_for_actual_catastrophic']['auc'])}。**解释：**条件top-1好主要集中在少量严格正事件，跨全部事件的绝对分数仍未校准。**含义：**下一版要把“是否commit”和“commit哪个candidate”拆成两个头。",
            f"5. **观察：current与last-reliable候选大量重复。** peak0输出完全相同{duplicate['peak0_identical']}/{duplicate['events']}个事件，peak1为{duplicate['peak1_identical']}/{duplicate['events']}。**解释：**六候选名义动作空间通常只有4个独立输出。**含义：**还必须去重或加入真正不同的re-detection/instance-memory候选。",
            f"6. **观察：predicted gain与真实H4 gain几乎无关。** Pearson={fmt(m7['gain_regression']['pearson'])}，Spearman={fmt(m7['gain_regression']['spearman'])}。**解释：**当前gain回归不能承担生存决策。**含义：**下一版应直接学习严格H10 commitability和严格action label，而不是继续依赖H4 gain门。",
            "",
            "## 下一步（不自动执行公开评测）",
            "",
            "1. 正式封存M7为负结果，不扫描阈值、epoch、loss权重，也不运行low22。",
            "2. 如果继续STTrack，只允许M8实质重构：以Gate-A严格H10标签为唯一监督，拆成`event commitability`与`conditional candidate ranking`两阶段；不再用H4宽松标签决定abstain。",
            "3. M8同时对current/last重复假设做显式去重，并加入真正不同的last-reliable/global re-detection或target/distractor memory候选；先做Train-only容量门。",
            "4. 更强baseline迁移仍须满足源码、对应权重、官方VOT三指标和许可证。MDTrack/FlexTrack当前发布物不满足，FlexTrackV2只能保持探索状态。",
            "5. 任何新结构先用DepthTrack Train做sequence-disjoint容量与零灾难门；只有通过才允许low22。",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m7-eval", required=True)
    parser.add_argument("--m7-trace", required=True)
    parser.add_argument("--m6-eval", required=True)
    parser.add_argument("--m6-trace", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--md-output", required=True)
    args = parser.parse_args()

    report = {
        "schema": "sttrack-lachtt-m7-readonly-audit/v1",
        "gates": GATES,
        "m7": {
            "evaluation": event_summary(read_jsonl_gz(args.m7_eval)),
            "training": trace_summary(read_jsonl_gz(args.m7_trace)),
        },
        "m6": {
            "evaluation": event_summary(read_jsonl_gz(args.m6_eval)),
            "training": trace_summary(read_jsonl_gz(args.m6_trace)),
        },
        "public_benchmark_run": False,
        "threshold_scan": False,
        "tracking_checkpoint_written": False,
    }
    json_path = Path(args.json_output)
    md_path = Path(args.md_output)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "m7_selected": report["m7"]["evaluation"]["independent_gate_pass_counts"]["all"],
        "m7_events": report["m7"]["evaluation"]["events"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
