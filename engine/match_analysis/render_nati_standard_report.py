from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


def mock_payload() -> dict[str, Any]:
    """返回一份用于版式测试的假数据。

    字段比旧版更丰富，目的是贴近标准报告的信息密度。脚本仍兼容旧 JSON，
    缺失字段会在 normalize_payload() 中自动补齐。
    """
    return {
        "schema_version": "nati-assessment-input-v1",
        "data_status": "mock",
        "mock_notice": "演示假数据，仅用于报告版式测试，不可作为真实球员评估依据。",
        "player": {
            "player_id": "white_26",
            "display_name": "小宇",
            "team": "白队",
            "jersey_number": 26,
            "age_group": "U12",
            "position": "中前卫 / 前腰",
            "preferred_foot": "右脚",
            "club": "NATI 测试队",
            "profile_no": "NO.026",
            "registry_status": "演示档案",
        },
        "match": {
            "report_id": "NATI-MOCK-20260820-026",
            "assessment_date": "2026-08-20",
            "tournament": "U12 教学赛（演示）",
            "season": "2026 夏季",
            "match_accolade": "本场评定 · 节奏核心",
        },
        "headline": {
            "nickname": "枢纽",
            "quote": "先观察，再接球；先创造角度，再向前推进。",
        },
        "key_metrics": [
            {
                "key": "duel_retention",
                "value": 87,
                "display": "87%",
                "label": "对抗护球成功率",
                "quote": "受压时先稳住身体，再把球交到安全侧。",
            },
            {
                "key": "scan_before_receive",
                "value": 6,
                "display": "×6",
                "label": "接球前有效观察",
                "quote": "球到脚前已经看过弱侧和身后空间。",
            },
            {
                "key": "progressive_actions",
                "value": 4,
                "display": "4",
                "label": "向前推进",
                "quote": "能把安全接球转化为下一步向前选择。",
            },
        ],
        "radar": {
            "dimensions": {"体": 72, "技": 81, "战": 76, "心": 74, "智": 79, "观": 86, "决": 77, "位": 83},
            "scale": [0, 100],
        },
        "overall": {
            "ca_score": 78,
            "potential_grade": "A-",
            "potential_direction": "向 A 进发",
        },
        "analysis": {
            "strengths": [
                "接球前多次观察，能提前发现弱侧空间",
                "受压时保持身体朝向，第一脚处理稳定",
            ],
            "improvements": [
                "丢球后前三步回追要更坚决",
                "向前传递后继续移动，形成二次接应角度",
            ],
            "strengths_summary": "观 86 · 位 83 · 技 81 —— 接球前观察充分，身体朝向稳定，能用位置选择帮助球队连续推进。",
            "improvements_summary": "心 74 · 决 77 —— 丢球后的第一反应和出球后的再次移动还可更坚决，提升连续参与度。",
        },
        "style_archetype": {
            "triangle": {"tactical_literacy": 84, "physical_competition": 68, "talent": 79},
            "reference_player": "佩德里",
            "style_tag": "空间连接型 · 风格相似 ≠ 水平相当",
            "narrative": "他的价值更多来自提前观察和站位，而不是连续带球解决问题。观 86 与位 83 说明他能在接球前找到下一步空间；若能把出球后的再次移动做得更主动，连接能力会更完整。",
            "similarities": ["观察频率高，善于在小范围内提供接应角度"],
            "differences": ["高强度对抗后的动作连续性仍需提高"],
            "next_target": "出球后再接应 · 位 83 → 88",
        },
        "position_recommendations": [
            {
                "rank": 1,
                "position": "中前卫 / 组织型中场",
                "fit": 88,
                "description": "观察 + 接应角度 + 稳定第一脚，适合承担中路连接任务",
                "verdict": "极高匹配",
            },
            {
                "rank": 2,
                "position": "前腰",
                "fit": 82,
                "description": "弱侧发现能力较好，需继续提升最后一传与连续前插",
                "verdict": "较好匹配",
            },
            {
                "rank": 3,
                "position": "边前卫",
                "fit": 74,
                "description": "可利用观察和接应能力参与边路配合，爆发冲击不是主要优势",
                "verdict": "可尝试",
            },
        ],
        "messages": {
            "to_player": "你已经会用观察帮助自己踢得更从容。下一步是每次把球向前传出后，不停在原地，马上再创造一个新的接应角度。",
            "to_family_and_coach": "他的观察和位置感是当前最清晰的优势。建议训练中强化扫描-接球-转移-再接应的连续任务，并加入丢球后 3 秒反应规则，8-12 周后复评。",
        },
        "evidence": {
            "event_count": 6,
            "highlight_count": 3,
            "heatmap": "white_26/heatmap.png",
            "qr_caption": "扫码看本场\n高光片段",
        },
    }


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _fmt_date(value: str) -> str:
    value = str(value or "")
    if len(value) >= 10 and value[4] == "-" and value[7] == "-":
        return value[:10].replace("-", ".")
    return value


def _empty_payload() -> dict[str, Any]:
    """Return structural defaults only; never inject mock assessment values."""
    return {
        "schema_version": "nati-assessment-input-v1",
        "data_status": "evaluation_pending",
        "player": {},
        "match": {},
        "headline": {},
        "key_metrics": [],
        "radar": {"dimensions": {label: None for label in "体技战心智观决位"}, "scale": [0, 100]},
        "overall": {"ca_score": None, "potential_grade": None, "potential_direction": None},
        "analysis": {"strengths": [], "improvements": []},
        "style_archetype": {
            "triangle": {"tactical_literacy": None, "physical_competition": None, "talent": None},
            "reference_player": None,
            "similarities": [],
            "differences": [],
            "next_target": None,
        },
        "position_recommendations": [],
        "messages": {"to_player": None, "to_family_and_coach": None},
        "evidence": {},
    }


def _number(value: Any, default: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _display_metric(metric: dict[str, Any]) -> str:
    if metric.get("display") not in (None, ""):
        return str(metric["display"])
    value = _number(metric.get("value"))
    if value is None:
        return "待补充"
    key = str(metric.get("key", ""))
    if key == "total_distance_m":
        return f"{value:.1f} m"
    if key in {"max_speed_ms", "max_speed_mps"}:
        return f"{value:.2f} m/s"
    if key == "sprint_count":
        return str(int(round(value)))
    return f"{value:g}"


def normalize_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Adapt v1/v2 input to official v1 without inventing assessment values."""
    if not isinstance(raw, dict):
        raise TypeError("报告输入必须是 JSON object")
    source_version = raw.get("schema_version", "nati-assessment-input-v1")
    if source_version not in {"nati-assessment-input-v1", "nati-assessment-input-v2"}:
        raise ValueError(f"不支持的 schema_version: {source_version}")
    status = raw.get("data_status", "evaluation_pending")
    if status not in {"mock", "partial", "evaluation_pending", "final"}:
        raise ValueError(f"不支持的 data_status: {status}")

    p = _deep_merge(_empty_payload(), raw)
    p["schema_version"] = "nati-assessment-input-v1"
    if source_version != p["schema_version"]:
        p["source_schema_version"] = source_version

    player = p["player"]
    match = p["match"]
    analysis = p["analysis"]
    style = p["style_archetype"]

    is_pending = status in {"partial", "evaluation_pending"}
    jersey = int(_number(player.get("jersey_number"), 0) or 0)
    player["team"] = player.get("team") or "待补充队伍"
    player["age_group"] = player.get("age_group") or "待补充"
    player["position"] = player.get("position") or "待评估位置"
    player["club"] = player.get("club") or match.get("tournament") or "待补充俱乐部"
    player["profile_no"] = player.get("profile_no") or f"NO.{jersey:03d}"
    player["registry_status"] = player.get("registry_status") or (
        "演示档案" if status == "mock" else "评估待完成" if is_pending else "已入册"
    )

    match["report_id"] = match.get("report_id") or f"NATI-PENDING-{player.get('player_id') or jersey}"
    match["assessment_date"] = match.get("assessment_date") or "待评估"
    match["season"] = match.get("season") or "待补充"
    match["match_accolade"] = match.get("match_accolade") or (
        "本场评定 · 数据待评估" if is_pending else "本场评定"
    )
    p["headline"]["nickname"] = p["headline"].get("nickname") or ("待评估" if is_pending else "本场球员")
    p["headline"]["quote"] = p["headline"].get("quote") or (
        "语义评估尚未完成；当前仅展示已确认的客观数据。" if is_pending else "评估内容待补充。"
    )

    # 旧版的 potential_direction 更像球员类型，标准版这一格更适合显示“向 A- 进发”。
    if not p["overall"].get("potential_direction"):
        grade = p["overall"].get("potential_grade")
        p["overall"]["potential_direction"] = f"向 {grade} 进发" if grade else "等待评估数据"

    dims = p["radar"].get("dimensions", {})
    numeric_dims = [(key, value) for key, value in dims.items() if _number(value) is not None]
    if not analysis.get("strengths_summary"):
        top = sorted(numeric_dims, key=lambda kv: float(kv[1]), reverse=True)[:4]
        anchors = " · ".join(f"{k} {v}" for k, v in top)
        body = "、".join(analysis.get("strengths", []))
        analysis["strengths_summary"] = (
            f"{anchors} —— {body}。" if anchors and body else body or anchors or "等待语义评估结果，不以跑动数据推断技术能力。"
        )
    if not analysis.get("improvements_summary"):
        low = sorted(numeric_dims, key=lambda kv: float(kv[1]))[:2]
        anchors = " · ".join(f"{k} {v}" for k, v in low)
        body = "、".join(analysis.get("improvements", []))
        analysis["improvements_summary"] = (
            f"{anchors} —— {body}。" if anchors and body else body or anchors or "待结合事件标注与视频复核后给出训练建议。"
        )

    if not style.get("style_tag"):
        style["style_tag"] = "风格相似 ≠ 水平相当"
    if not style.get("narrative"):
        strengths = "、".join(analysis.get("strengths", []))
        style["narrative"] = (
            f"本场已确认特征：{strengths}。下一步需要把优势稳定地转化成更多连续参与。"
            if strengths else "风格原型需要语义事件与人工复核，当前不根据跑动指标自动推断。"
        )
    style["reference_player"] = style.get("reference_player") or "待评估"
    style["next_target"] = style.get("next_target") or "待评估完成后生成"

    # 旧版三项 KPI 没有 quote；补齐后视觉更接近标准版。
    default_quotes = [
        "来自自动汇总，需结合视频复核。",
        "本场客观统计，不代表长期能力。",
        "数据口径见随报告交付的接口文件。",
    ]
    for idx, metric in enumerate(p.get("key_metrics", [])[:3]):
        metric["display"] = _display_metric(metric)
        metric["quote"] = metric.get("quote") or default_quotes[idx]
    p["key_metrics"] = p.get("key_metrics", [])[:3]
    while len(p["key_metrics"]) < 3:
        i = len(p["key_metrics"])
        p["key_metrics"].append({
            "key": f"pending_metric_{i + 1}", "value": None, "display": "待补充",
            "label": "客观指标待补充", "quote": "不会使用假数据填充正式报告。",
        })

    for idx, row in enumerate(p.get("position_recommendations", [])[:3]):
        row.setdefault("rank", idx + 1)
        row.setdefault("description", "基于本场行为特征的测试版位置适配描述")
        fit = _number(row.get("fit"))
        row.setdefault("verdict", "待评估" if fit is None else "极高匹配" if fit >= 86 else "较好匹配" if fit >= 72 else "谨慎匹配")
    p["position_recommendations"] = p.get("position_recommendations", [])[:3]
    while len(p["position_recommendations"]) < 3:
        i = len(p["position_recommendations"])
        p["position_recommendations"].append({
            "rank": i + 1, "position": "待评估", "fit": None,
            "description": "等待语义事件、视频复核与评估数据", "verdict": "待评估",
        })

    p["messages"]["to_player"] = p["messages"].get("to_player") or "评估尚未完成。当前客观数据已入库，待完成视频事件复核后生成个性化建议。"
    p["messages"]["to_family_and_coach"] = p["messages"].get("to_family_and_coach") or "跑动数据不直接等同于技术、战术或潜力评分；正式结论需结合事件标注与人工复核。"

    match["assessment_date_display"] = _fmt_date(match.get("assessment_date", ""))
    return p


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _radar_svg(scores: dict[str, Any], size: int = 205) -> str:
    labels = list(scores.keys())[:8]
    while len(labels) < 8:
        labels.append(f"维{len(labels)+1}")
    raw_values = [_number(scores.get(label)) for label in labels]
    complete = all(value is not None for value in raw_values)
    values = [max(0.0, min(100.0, value or 0.0)) for value in raw_values]
    cx = cy = size / 2
    radius = 66
    angles = [math.radians(-90 + i * 45) for i in range(8)]

    def points(scale: float) -> str:
        return " ".join(
            f"{cx + radius * scale * math.cos(a):.1f},{cy + radius * scale * math.sin(a):.1f}"
            for a in angles
        )

    data_pts = " ".join(
        f"{cx + radius * (v/100) * math.cos(a):.1f},{cy + radius * (v/100) * math.sin(a):.1f}"
        for v, a in zip(values, angles)
    )
    grid = "".join(f'<polygon points="{points(s)}" class="radar-grid"/>' for s in (0.25, 0.5, 0.75, 1.0))
    axes = "".join(
        f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{cx + radius*math.cos(a):.1f}" y2="{cy + radius*math.sin(a):.1f}" class="radar-axis"/>'
        for a in angles
    )
    labels_svg = []
    for label, value, raw_value, a in zip(labels, values, raw_values, angles):
        rr = radius + 18
        x = cx + rr * math.cos(a)
        y = cy + rr * math.sin(a)
        anchor = "middle"
        if math.cos(a) > 0.35:
            anchor = "start"
        elif math.cos(a) < -0.35:
            anchor = "end"
        shown = str(int(round(value))) if raw_value is not None else "-"
        cls = " radar-low" if complete and value == min(values) else ""
        labels_svg.append(
            f'<text x="{x:.1f}" y="{y+3:.1f}" text-anchor="{anchor}" class="radar-label{cls}">{_esc(label)} {shown}</text>'
        )
    if complete:
        data_layer = (
            f'<polygon points="{data_pts}" class="radar-area"/>'
            + ''.join(
                f'<circle cx="{cx + radius*(v/100)*math.cos(a):.1f}" cy="{cy + radius*(v/100)*math.sin(a):.1f}" r="2.7" class="radar-dot"/>'
                for v, a in zip(values, angles)
            )
        )
    else:
        data_layer = f'<text x="{cx:.1f}" y="{cy+3:.1f}" text-anchor="middle" class="radar-pending">待评估</text>'
    return f"""
    <svg viewBox="0 0 {size} {size}" class="radar-svg" aria-label="八维能力雷达">
      <style>
        .radar-grid{{fill:none;stroke:#d9dfe5;stroke-width:1}}
        .radar-axis{{stroke:#e5e9ed;stroke-width:1}}
        .radar-area{{fill:rgba(0,160,116,.18);stroke:#00a476;stroke-width:2.1}}
        .radar-dot{{fill:#00a476}}
        .radar-label{{font:700 10px 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#1d293b}}
        .radar-low{{fill:#e2454c}}
        .radar-pending{{font:700 10px 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#aab3bb}}
      </style>
      {grid}{axes}
      {data_layer}
      {''.join(labels_svg)}
    </svg>
    """


def _triangle_svg(tri: dict[str, Any]) -> str:
    raw_values = [_number(tri.get(key)) for key in ("tactical_literacy", "physical_competition", "talent")]
    complete = all(value is not None for value in raw_values)
    tactical, physical, talent = [value if value is not None else 50.0 for value in raw_values]
    # 只用于风格定位的视觉投影，不表示数学意义上的能力合成。
    px = 72 + (talent - physical) * 0.48
    py = 72 - (tactical - 50) * 0.72
    px = max(40, min(104, px))
    py = max(34, min(92, py))
    return f"""
    <svg viewBox="0 0 145 128" class="tri-svg" aria-label="风格原型三角定位">
      <polygon points="72,18 20,106 124,106" fill="#fafbfc" stroke="#d7dde2" stroke-width="1.2"/>
      <polygon points="72,46 43,92 101,92" fill="none" stroke="#e8ecef" stroke-width="1"/>
      <text x="72" y="13" text-anchor="middle" class="tri-axis">战术素养</text>
      <text x="8" y="118" text-anchor="start" class="tri-axis">身体强悍</text>
      <text x="137" y="118" text-anchor="end" class="tri-axis">天赋</text>
      <circle cx="72" cy="29" r="3" fill="#cfd6dc"/><text x="79" y="32" class="tri-ghost">穆勒</text>
      {f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4.4" fill="#00a476"/><text x="{px-7:.1f}" y="{py-8:.1f}" text-anchor="middle" class="tri-player">球员</text>' if complete else '<text x="72" y="70" text-anchor="middle" class="tri-pending">待评估</text>'}
      <style>
        .tri-axis{{font:700 8px 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#202b3c}}
        .tri-ghost{{font:400 6px 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#aab3bb}}
        .tri-player{{font:700 7px 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#00a476}}
        .tri-pending{{font:700 8px 'Noto Sans CJK SC','Microsoft YaHei',sans-serif;fill:#aab3bb}}
      </style>
    </svg>
    """


def _kpi_html(metric: dict[str, Any]) -> str:
    display = _esc(metric.get("display", metric.get("value", "-")))
    label = _esc(metric.get("label", "关键行为"))
    quote = _esc(metric.get("quote", ""))
    return f"""
      <div class="kpi-card">
        <div class="kpi-value">{display}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-quote">“{quote}”</div>
      </div>
    """


def build_html(payload: dict[str, Any]) -> str:
    p = payload
    player = p["player"]
    match = p["match"]
    headline = p["headline"]
    overall = p["overall"]
    analysis = p["analysis"]
    style = p["style_archetype"]
    evidence = p.get("evidence", {})
    dims = p["radar"]["dimensions"]
    nickname_text = str(headline.get("nickname", ""))
    nickname_len = len(nickname_text)
    nickname_size = 48 if nickname_len <= 2 else 40 if nickname_len <= 4 else 31
    nickname_spacing = 10 if nickname_len <= 2 else 5 if nickname_len <= 4 else 2

    metrics = "".join(_kpi_html(m) for m in p["key_metrics"][:3])
    radar = _radar_svg(dims)
    triangle = _triangle_svg(style.get("triangle", {}))
    ca_display = overall.get("ca_score") if overall.get("ca_score") is not None else "待评估"
    potential_display = overall.get("potential_grade") or "待评估"

    similarities = "<br>".join(_esc(x) for x in style.get("similarities", [])) or "-"
    differences = "<br>".join(_esc(x) for x in style.get("differences", [])) or "-"

    pos_rows = []
    for row in p["position_recommendations"][:3]:
        fit_value = _number(row.get("fit"))
        fit = int(round(fit_value)) if fit_value is not None else None
        fit_display = str(fit) if fit is not None else "--"
        tone = "pending" if fit is None else "high" if fit >= 86 else "mid" if fit >= 72 else "low"
        pos_rows.append(f"""
        <div class="position-row {tone}">
          <div class="fitbox"><div class="fitnum">{fit_display}</div><div class="fitcap">风格匹配度</div></div>
          <div class="posname">{_esc(row.get('position','-'))}</div>
          <div class="posdesc">{_esc(row.get('description',''))}</div>
          <div class="verdict">{_esc(row.get('verdict',''))}</div>
        </div>
        """)

    disclaimer = (
        "称号逐场评定，由 NATI 球探团队根据本场行为数据授予 · 风格原型为风格参照，不代表水平等同 · "
        "百分制为行为锚定评分，参照 U12 青训比赛场景 · 完整数据锚点与事件台账备索"
    )
    if p.get("data_status") == "mock":
        disclaimer = f"{p.get('mock_notice','演示假数据')} · {disclaimer}"

    # 参考标准版：内容区窄、留白大、全页白底、绿色强调、深蓝标题、金色评定。
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>NATI 球员评估报告 · {_esc(player.get('team'))}{_esc(player.get('jersey_number'))}号</title>
<style>
  @page {{ size: A4; margin: 0; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #fff; }}
  body {{
    font-family: "Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
    color: #162238;
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
  }}
  .page {{ width: 793.7px; height: 1122.5px; position: relative; overflow: hidden; background: #fff; margin: 0 auto; }}
  .report {{ width: 540px; margin: 18px auto 0; }}
  .header {{ display:flex; justify-content:space-between; align-items:flex-start; padding:0 2px 10px; border-bottom:3px solid #18253a; }}
  .brand {{ display:flex; align-items:baseline; white-space:nowrap; }}
  .nati {{ color:#00a476; font-weight:800; font-size:20px; letter-spacing:.5px; }}
  .cnbrand {{ color:#142037; font-weight:800; font-size:15px; margin-left:9px; }}
  .brand-sep {{ color:#adb4bc; margin:0 8px; font-size:10px; }}
  .eng {{ font-size:7px; letter-spacing:4px; color:#46505c; font-weight:500; }}
  .meta {{ text-align:right; font-size:7.3px; line-height:1.55; color:#313844; margin-top:1px; }}
  .meta b {{ color:#172237; }}

  .hero {{ text-align:center; padding-top:11px; padding-bottom:11px; }}
  .eyebrow {{ color:#b98c00; font-weight:800; font-size:8px; letter-spacing:7px; margin-bottom:2px; }}
  .nickname {{ font-size:48px; line-height:1.05; font-weight:900; letter-spacing:10px; margin-left:10px; color:#142038; }}
  .quote {{ color:#00a476; font-size:13.2px; font-weight:800; margin-top:3px; }}
  .identity {{ font-size:8.5px; color:#46505e; margin-top:6px; font-weight:600; letter-spacing:.3px; }}
  .accolade {{ display:inline-block; margin-top:7px; padding:5px 18px 4px; border:1.6px solid #d3a20b; color:#b68a00; border-radius:20px; font-size:8.7px; font-weight:800; letter-spacing:4px; }}

  .kpis {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; margin-top:2px; }}
  .kpi-card {{ height:83px; border:1px solid #dfe4e8; border-top:3px solid #00a476; border-radius:9px; text-align:center; padding:8px 8px 6px; box-shadow:0 1px 2px rgba(0,0,0,.02); }}
  .kpi-value {{ color:#00a476; font-weight:850; font-size:27px; line-height:1; }}
  .kpi-label {{ color:#1b2538; font-size:9px; font-weight:800; margin-top:5px; }}
  .kpi-quote {{ color:#9aa2ac; font-size:7px; font-style:italic; margin-top:5px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}

  .assessment {{ display:grid; grid-template-columns:230px 1fr; gap:12px; margin-top:12px; align-items:stretch; }}
  .card {{ border:1px solid #dfe4e8; border-radius:9px; background:#fff; }}
  .radar-card {{ height:238px; padding:10px 12px 7px; text-align:center; overflow:hidden; }}
  .card-title {{ font-size:10px; font-weight:850; margin-bottom:2px; }}
  .radar-dims {{ color:#9aa2ac; font-size:6.4px; letter-spacing:3.6px; margin-bottom:1px; white-space:nowrap; }}
  .radar-svg {{ width:205px; height:196px; display:block; margin:-2px auto 0; }}
  .right-assess {{ display:flex; flex-direction:column; gap:8px; }}
  .score-row {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; height:52px; }}
  .score-card {{ border:1px solid #dfe4e8; border-radius:9px; text-align:center; padding:6px 4px 4px; }}
  .score-card.ca {{ border-color:#d4a10a; }}
  .score-big {{ font-size:26px; font-weight:850; line-height:1; }}
  .score-card.ca .score-big {{ color:#c7990a; }}
  .score-card.potential .score-big {{ color:#1a2233; }}
  .score-caption {{ color:#9da4ad; font-size:6.9px; margin-top:3px; }}
  .analysis-card {{ height:64px; padding:8px 11px 7px; position:relative; overflow:hidden; }}
  .analysis-card:before {{ content:""; position:absolute; left:0; top:0; bottom:0; width:3px; border-radius:9px 0 0 9px; }}
  .analysis-card.good:before {{ background:#00a476; }}
  .analysis-card.work:before {{ background:#e54752; }}
  .analysis-label {{ font-size:8px; font-weight:850; margin-bottom:4px; }}
  .good .analysis-label {{ color:#00a476; }}
  .work .analysis-label {{ color:#e54752; }}
  .analysis-text {{ font-size:7.8px; line-height:1.55; color:#313b49; }}
  .analysis-text b {{ color:#121d31; }}

  .section-head {{ display:flex; align-items:center; justify-content:space-between; margin:12px 0 6px; }}
  .section-title {{ border-left:4px solid #00a476; padding-left:8px; font-size:11px; font-weight:900; line-height:15px; }}
  .section-en {{ color:#aeb4bc; font-size:6.2px; letter-spacing:3px; }}

  .style-grid {{ display:grid; grid-template-columns:118px 164px 148px 92px; gap:8px; height:143px; }}
  .style-card {{ border:1px solid #dfe4e8; border-radius:9px; padding:8px; overflow:hidden; }}
  .tri-card {{ padding:6px 4px 2px; }}
  .tri-svg {{ width:110px; height:112px; display:block; margin:0 auto; }}
  .style-main {{ border-top:3px solid #17243a; padding:8px 9px 6px; }}
  .style-ref {{ font-size:10px; font-weight:900; margin-bottom:3px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .style-tag {{ color:#a4aab2; font-size:6.2px; margin-bottom:5px; }}
  .style-narrative {{ font-size:7.5px; line-height:1.52; color:#303947; }}
  .style-narrative b {{ font-weight:850; color:#142039; }}
  .compare-card {{ font-size:7.1px; line-height:1.45; color:#343d48; }}
  .compare-label {{ font-weight:850; margin-bottom:2px; }}
  .compare-label.sim {{ color:#00a476; }}
  .compare-label.diff {{ color:#e54752; margin-top:5px; }}
  .compare-label.target {{ color:#b88c00; margin-top:5px; }}
  .qr-card {{ text-align:center; color:#8f98a3; font-size:7px; padding-top:14px; }}
  .qrbox {{ width:58px; height:58px; border:1.4px dashed #b8c0c8; border-radius:7px; margin:0 auto 8px; display:flex; align-items:center; justify-content:center; color:#bcc3c9; line-height:1.4; font-size:7px; }}

  .positions {{ display:flex; flex-direction:column; gap:6px; }}
  .position-row {{ height:43px; border:1px solid #e0e4e8; border-radius:8px; display:grid; grid-template-columns:48px 120px 1fr 58px; align-items:center; padding:0 10px 0 8px; }}
  .fitbox {{ text-align:center; line-height:1; }}
  .fitnum {{ font-size:16px; font-weight:850; }}
  .fitcap {{ color:#aab1ba; font-size:5.3px; letter-spacing:1.2px; margin-top:3px; }}
  .high .fitnum, .high .verdict {{ color:#00a476; }}
  .mid .fitnum {{ color:#596475; }}
  .low .fitnum, .low .verdict {{ color:#b9c0c7; }}
  .pending .fitnum, .pending .verdict {{ color:#aab1ba; }}
  .posname {{ font-size:9.2px; font-weight:850; color:#152035; }}
  .posdesc {{ font-size:7px; color:#4e5864; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; padding-right:6px; }}
  .verdict {{ text-align:right; font-size:7px; font-weight:800; color:#66717d; }}

  .messages {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }}
  .message {{ height:80px; border-radius:9px; padding:10px 12px; font-size:8.2px; line-height:1.6; color:#34404d; overflow:hidden; }}
  .message.player {{ background:#e9f7f1; }}
  .message.coach {{ background:#f4f5f7; }}
  .message-title {{ font-size:8.4px; font-weight:850; margin-bottom:3px; }}
  .message.player .message-title {{ color:#00a476; }}
  .message.coach .message-title {{ color:#434d5d; }}
  .message b {{ color:#172239; font-weight:850; }}

  .footer {{ border-top:1px solid #e1e5e8; margin-top:10px; padding-top:7px; display:flex; justify-content:space-between; gap:16px; }}
  .disclaimer {{ width:410px; color:#b0b6bd; font-size:5.7px; line-height:1.55; }}
  .footer-brand {{ text-align:right; white-space:nowrap; }}
  .footer-brand .ai {{ font-size:8px; font-weight:900; letter-spacing:1.5px; color:#1b2538; }}
  .footer-brand .sub {{ color:#9ba3ad; font-size:5.8px; margin-top:3px; }}
</style>
</head>
<body>
<div class="page">
  <main class="report">
    <header class="header">
      <div class="brand"><span class="nati">NATI</span><span class="cnbrand">哪踢</span><span class="brand-sep">|</span><span class="eng">球员正式评估报告</span></div>
      <div class="meta">
        报告编号 <b>#{_esc(match.get('report_id'))}</b> · 评估周期 <b>{_esc(match.get('assessment_date_display'))}</b><br>
        球员档案 <b>{_esc(player.get('profile_no'))}</b> · {_esc(player.get('registry_status'))} · 赛季卡 {_esc(match.get('season'))}
      </div>
    </header>

    <section class="hero">
      <div class="eyebrow">本 场 称 号</div>
      <div class="nickname" style="font-size:{nickname_size}px;letter-spacing:{nickname_spacing}px;margin-left:{nickname_spacing}px">{_esc(headline.get('nickname'))}</div>
      <div class="quote">“{_esc(headline.get('quote'))}”</div>
      <div class="identity">{_esc(player.get('team'))} {_esc(player.get('jersey_number'))} 号 · {_esc(player.get('age_group'))} · {_esc(player.get('position'))} · {_esc(player.get('preferred_foot') or '惯用脚待补充')} · {_esc(player.get('club'))}</div>
      <div class="accolade">{_esc(match.get('match_accolade'))}</div>
    </section>

    <section class="kpis">{metrics}</section>

    <section class="assessment">
      <div class="card radar-card">
        <div class="card-title">八维能力雷达</div>
        <div class="radar-dims">体 · 技 · 战 · 心 · 智 · 观 · 决 · 位</div>
        {radar}
      </div>
      <div class="right-assess">
        <div class="score-row">
          <div class="score-card ca"><div class="score-big">{_esc(ca_display)}</div><div class="score-caption">综合评分</div></div>
          <div class="score-card potential"><div class="score-big">{_esc(potential_display)}</div><div class="score-caption">潜力 · {_esc(overall.get('potential_direction'))}</div></div>
        </div>
        <div class="card analysis-card good"><div class="analysis-label">优势区</div><div class="analysis-text">{_esc(analysis.get('strengths_summary'))}</div></div>
        <div class="card analysis-card work"><div class="analysis-label">攻坚区</div><div class="analysis-text">{_esc(analysis.get('improvements_summary'))}</div></div>
      </div>
    </section>

    <div class="section-head"><div class="section-title">风格原型 · 三角定位</div><div class="section-en">球员风格定位</div></div>
    <section class="style-grid">
      <div class="style-card tri-card">{triangle}</div>
      <div class="style-card style-main">
        <div class="style-ref">原型参照：{_esc(style.get('reference_player'))}</div>
        <div class="style-tag">{_esc(style.get('style_tag'))}</div>
        <div class="style-narrative">{_esc(style.get('narrative'))}</div>
      </div>
      <div class="style-card compare-card">
        <div class="compare-label sim">▲ 相似点</div><div>{similarities}</div>
        <div class="compare-label diff">▼ 差异点</div><div>{differences}</div>
        <div class="compare-label target">◎ 本赛季解锁目标</div><div>{_esc(style.get('next_target'))}</div>
      </div>
      <div class="style-card qr-card">
        <div class="qrbox">二维码<br>预留位</div>
        {_esc(evidence.get('qr_caption','扫码看本场高光片段')).replace(chr(10), '<br>')}
      </div>
    </section>

    <div class="section-head"><div class="section-title">推荐发展位置</div><div class="section-en">BEST FIT · TOP 3</div></div>
    <section class="positions">{''.join(pos_rows)}</section>

    <section class="messages">
      <div class="message player"><div class="message-title">给孩子</div>{_esc(p['messages'].get('to_player'))}</div>
      <div class="message coach"><div class="message-title">给家长 & 教练</div>{_esc(p['messages'].get('to_family_and_coach'))}</div>
    </section>

    <footer class="footer">
      <div class="disclaimer">{_esc(disclaimer)}</div>
      <div class="footer-brand"><div class="ai">NATI · 智能球探</div><div class="sub">哪踢科技 · 青少年球员成长评估系统</div></div>
    </footer>
  </main>
</div>
</body>
</html>"""


def find_chrome(explicit: str | None = None) -> str:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        found = shutil.which(explicit)
        if found:
            return found
        raise FileNotFoundError(f"找不到指定浏览器: {explicit}")

    env = os.environ.get("NATI_CHROME") or os.environ.get("CHROME_BIN")
    if env and Path(env).exists():
        return env

    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "msedge", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in DEFAULT_CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "未找到 Chrome/Chromium/Edge。请安装浏览器，或通过 --chrome / NATI_CHROME 指定可执行文件。"
    )


def render_pdf(html_text: str, output: Path, chrome: str | None = None, keep_html: Path | None = None) -> None:
    """优先通过 Playwright 控制本机 Chrome 打印 PDF；失败时再退回命令行模式。"""
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    browser_path = find_chrome(chrome)

    if keep_html:
        keep_html.parent.mkdir(parents=True, exist_ok=True)
        keep_html.write_text(html_text, encoding="utf-8")

    playwright_error: Exception | None = None
    try:
        from playwright.sync_api import sync_playwright  # type: ignore

        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                executable_path=browser_path,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            page = browser.new_page()
            # set_content 比 file:// 导航更稳，也避免某些企业环境拦截本地文件 URL。
            page.set_content(html_text, wait_until="load", timeout=30_000)
            page.pdf(
                path=str(output),
                format="A4",
                print_background=True,
                prefer_css_page_size=True,
                margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            )
            browser.close()
        return
    except Exception as exc:
        playwright_error = exc

    with tempfile.TemporaryDirectory(prefix="nati_report_") as td:
        html_path = Path(td) / "report.html"
        html_path.write_text(html_text, encoding="utf-8")
        cmd = [
            browser_path,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--print-to-pdf-no-header",
            f"--print-to-pdf={output}",
            html_path.resolve().as_uri(),
        ]
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                encoding="utf-8", errors="replace", timeout=45,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "生成 PDF 失败：Playwright 与 Chrome 命令行两种方式都没有成功。\n"
                f"Playwright error: {playwright_error}\nChrome timeout: {exc}"
            ) from exc
        if proc.returncode != 0 or not output.exists():
            raise RuntimeError(
                "浏览器生成 PDF 失败。\n"
                f"Playwright error: {playwright_error}\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout: {proc.stdout}\n"
                f"stderr: {proc.stderr}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成接近 NATI 标准版视觉的一页球员评估报告")
    parser.add_argument("--input-json", type=Path, help="输入 JSON；不传则使用内置 mock 数据")
    parser.add_argument("--output", type=Path, required=True, help="输出 PDF")
    parser.add_argument("--json-output", type=Path, help="可选：输出规范化后的 JSON")
    parser.add_argument("--html-output", type=Path, help="可选：保留中间 HTML，便于调样式")
    parser.add_argument("--chrome", help="Chrome/Chromium/Edge 可执行文件路径")
    parser.add_argument("--force", action="store_true", help="允许覆盖输出文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for target in (args.output, args.json_output, args.html_output):
        if target and target.exists() and not args.force:
            raise FileExistsError(f"refusing to overwrite: {target}; 使用 --force 覆盖")

    if args.input_json:
        raw = json.loads(args.input_json.read_text(encoding="utf-8"))
    else:
        raw = mock_payload()
    payload = normalize_payload(raw)

    html_text = build_html(payload)
    render_pdf(html_text, args.output, args.chrome, args.html_output)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "pdf": str(args.output.resolve()),
        "json": str(args.json_output.resolve()) if args.json_output else None,
        "html": str(args.html_output.resolve()) if args.html_output else None,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
