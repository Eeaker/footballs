from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path
from typing import Any


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open("r", encoding="utf-8-sig", newline="") as f: return list(csv.DictReader(f))


def _f(value, default=0.0):
    try: return float(value)
    except Exception: return default


def _i(value, default=0):
    try: return int(float(value))
    except Exception: return default


def _fmt_time(sec: float) -> str:
    sec=max(0,float(sec)); return f"{int(sec//60):02d}:{int(sec%60):02d}"


def build_match_report(project: dict[str, Any], output_root: Path) -> tuple[Path, Path | None]:
    """Generate a self-contained, printable multi-section product report."""
    match_analysis = output_root / "match_analysis"
    running = _csv(match_analysis / "metric_running" / "player_running_summary.csv")
    passes = _csv(match_analysis / "analysis" / "pass_events.csv")
    possessions = _csv(match_analysis / "analysis" / "possession_intervals.csv")
    teams = _csv(match_analysis / "analysis" / "team_pass_summary.csv")
    team_map = {r.get("global_id"): r.get("team_id") for r in _csv(match_analysis / "analysis" / "player_team_map.csv")}
    cards = _csv(output_root / "player_cards" / "player_running_summary.csv")
    numbers = {r.get("global_id"): r for r in _csv(output_root / "number_ocr" / "jersey_number_results.csv")}
    match = project.get("match") or {}
    video = project.get("video") or {}
    identity_map = {}
    assessment_map = {}
    pass_review_state = {"labeled": 0, "total": 0, "agreement_rate": None}
    if project.get("kind") != "demo":
        try:
            from app.services.reviews import identity_mapping_dict, load_pass_review, player_assessment_dict
            identity_map = identity_mapping_dict(project)
            assessment_map = player_assessment_dict(project)
            pass_review_state = load_pass_review(project)
        except Exception:
            identity_map = {}
            assessment_map = {}
    def identity_for_gid(gid: str | int) -> dict[str, Any]:
        return identity_map.get(str(gid), {}) if identity_map else {}
    def assessment_for_gid(gid: str | int) -> dict[str, Any]:
        return assessment_map.get(str(gid), {}) if assessment_map else {}
    def assessment_text(gids: list[str]) -> str:
        a = next((assessment_for_gid(g) for g in gids if assessment_for_gid(g)), {})
        scores = a.get("scores", {}) if isinstance(a, dict) else {}
        if a.get("status") != "confirmed" or len(scores) != 8:
            return "待评估"
        try:
            avg = sum(float(v) for v in scores.values()) / 8
            return f"已确认 · {avg:.0f}"
        except Exception:
            return "已确认"
    peak = max((_f(r.get("peak_speed_mps_p95")) for r in running), default=0)
    total = sum(_f(r.get("total_distance_m")) for r in running)
    team_labels = match.get("team_labels") or {}
    def team_label(tid: str) -> str:
        return team_labels.get(tid) or {"team_0":"队伍 A","team_1":"队伍 B","team_2":"裁判/其他"}.get(tid, tid or "未分组")

    # Possession share from stable possession intervals.
    possession_frames = {}
    for r in possessions:
        tid=r.get("team_id"); possession_frames[tid]=possession_frames.get(tid,0)+max(0,_i(r.get("end_frame_proc"))-_i(r.get("start_frame_proc"))+1)
    total_pos=sum(possession_frames.values()) or 1
    team_cards=[]
    for r in teams:
        tid=r.get("team_id") or ""
        team_cards.append(f'''<div class="teamcard"><div class="teamname">{html.escape(team_label(tid))}</div><div class="teamgrid"><div><b>{_i(r.get('active_directed_passes'))}</b><span>传球候选</span></div><div><b>{_f(r.get('mean_pass_distance_m')):.1f}m</b><span>平均传球距离</span></div><div><b>{100*possession_frames.get(tid,0)/total_pos:.0f}%</b><span>稳定球权占比</span></div></div></div>''')

    player_rows=[]
    if cards:
        for r in cards[:30]:
            gids=[x.strip() for x in str(r.get("metric_global_ids") or "").replace(";",",").split(",") if x.strip()]
            ident=next((identity_for_gid(g) for g in gids if identity_for_gid(g)), {})
            player_rows.append({
                "name": ident.get("name") or r.get("player_id") or "球员", "number": ident.get("jersey_number") or r.get("jersey_number") or "—",
                "team": team_label(ident.get("team_id")) if ident.get("team_id") else (r.get("team") or ""),
                "distance": _f(r.get("total_distance")), "sprints": _i(r.get("sprint_count")), "speed": _f(r.get("max_speed_mps")), "quality": r.get("data_quality") or "",
                "assessment": assessment_text(gids),
            })
    else:
        for r in running[:30]:
            gid=str(r.get("global_id") or "") ; n=numbers.get(gid,{}) ; number=n.get("predicted_number") if "confirm" in str(n.get("status") or "").lower() else "待确认"
            ident=identity_for_gid(gid)
            tid=ident.get("team_id") or team_map.get(gid,"")
            player_rows.append({"name": ident.get("name") or f"ID {gid}", "number": ident.get("jersey_number") or number or "待确认", "team": team_label(tid), "distance": _f(r.get("total_distance_m")), "sprints": 0, "speed": _f(r.get("peak_speed_mps_p95")), "quality": r.get("quality_flags") or "", "assessment": assessment_text([gid])})
    player_rows.sort(key=lambda r:-r["distance"])
    player_html=''.join(f'''<tr><td><b>{html.escape(str(r['name']))}</b><small>{html.escape(str(r['team']))}</small></td><td>{html.escape(str(r['number']))}</td><td>{r['distance']/1000:.2f} km</td><td>{r['sprints'] if r['sprints'] else '—'}</td><td>{r['speed']:.2f} m/s</td><td>{html.escape(str(r.get('assessment','待评估')))}</td></tr>''' for r in player_rows[:20])

    def person_label(gid: Any) -> str:
        ident=identity_for_gid(str(_i(gid,-1)))
        return str(ident.get('name') or f"ID {_i(gid,-1)}")
    pass_html=''.join(f'''<tr><td>{_fmt_time(_f(r.get('release_time_seconds')))}</td><td>{html.escape(team_label(r.get('team_id') or ''))}</td><td>{html.escape(person_label(r.get('from_global_id')))} → {html.escape(person_label(r.get('to_global_id')))}</td><td>{_f(r.get('distance_m')):.1f} m</td><td>候选 · 可复核</td></tr>''' for r in passes[:20])
    cal=project.get("calibration") or {}; validation=cal.get("validation") or {}; coverage=validation.get("accepted_ratio")
    coverage_text=f"{float(coverage)*100:.1f}%" if coverage is not None else "—"

    doc=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(project.get('name','比赛分析报告'))}</title><style>
@page{{size:A4;margin:12mm}}*{{box-sizing:border-box}}:root{{--ink:#10213a;--muted:#718198;--line:#dfe7f1;--blue:#246bfe;--green:#1aaa76;--orange:#e58423;--panel:#f5f8fc}}body{{font-family:"Microsoft YaHei","PingFang SC",Arial,sans-serif;color:var(--ink);margin:0;background:#fff;font-size:11px;line-height:1.45}}header{{display:flex;justify-content:space-between;gap:20px;border-bottom:2px solid var(--blue);padding:4px 0 14px}}.brand{{font-weight:900;color:var(--blue);letter-spacing:.3px}}h1{{font-size:25px;margin:6px 0 3px}}h2{{font-size:15px;margin:0 0 10px}}.muted,small{{color:var(--muted)}}.meta{{text-align:right;line-height:1.7}}.metrics{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:14px 0}}.metric{{padding:11px;border-radius:10px;background:var(--panel);border:1px solid #e7edf5}}.metric span{{display:block;color:var(--muted);font-size:9px}}.metric b{{font-size:19px;display:block;margin-top:4px}}.section{{margin:16px 0;break-inside:avoid}}.teams{{display:grid;grid-template-columns:repeat(2,1fr);gap:9px}}.teamcard{{border:1px solid var(--line);border-radius:10px;padding:11px}}.teamname{{font-weight:800;margin-bottom:8px}}.teamgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}}.teamgrid div{{background:var(--panel);border-radius:7px;padding:7px}}.teamgrid b,.teamgrid span{{display:block}}.teamgrid b{{font-size:16px}}.teamgrid span{{font-size:8px;color:var(--muted)}}table{{width:100%;border-collapse:collapse;font-size:9px}}th,td{{padding:7px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:middle}}th{{color:#667891;background:var(--panel);font-weight:700}}td small{{display:block;font-size:7px;margin-top:2px}}.note{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}}.note div{{padding:10px;border:1px solid var(--line);border-radius:8px}}.note b,.note span{{display:block}}.note span{{font-size:8px;color:var(--muted);margin-top:3px}}footer{{margin-top:18px;padding-top:10px;border-top:1px solid var(--line);color:var(--muted);font-size:8px;display:flex;justify-content:space-between}}.pill{{display:inline-block;padding:3px 7px;border-radius:20px;background:#eaf1ff;color:#245bc5;font-size:8px;font-weight:800}}@media print{{.no-print{{display:none}}}}
</style></head><body>
<header><div><div class="brand">赛场洞察 · FOOTBALL INSIGHT</div><h1>{html.escape(project.get('name','比赛分析报告'))}</h1><div class="muted">{html.escape(match.get('competition') or '足球比赛视频智能分析')} · {html.escape(match.get('age_group') or '')}</div></div><div class="meta"><b>{html.escape(match.get('home_team') or '主队')} vs {html.escape(match.get('away_team') or '客队')}</b><br>{html.escape(match.get('match_date') or '')}<br>{html.escape(match.get('venue') or '')}</div></header>
<div class="metrics"><div class="metric"><span>候选球员轨迹</span><b>{len(running)}</b></div><div class="metric"><span>主动传球候选</span><b>{len(passes)}</b></div><div class="metric"><span>稳定球权片段</span><b>{len(possessions)}</b></div><div class="metric"><span>最高速度 P95</span><b>{peak:.2f}</b><small>m/s</small></div><div class="metric"><span>候选跑动总量</span><b>{total/1000:.1f}</b><small>km</small></div></div>
<section class="section"><h2>球队概览</h2><div class="teams">{''.join(team_cards) or '<div class="muted">暂无球队统计</div>'}</div></section>
<section class="section"><h2>球员表现</h2><table><thead><tr><th>球员</th><th>号码</th><th>跑动距离</th><th>冲刺</th><th>最高速度</th><th>八维评估</th></tr></thead><tbody>{player_html or '<tr><td colspan="6">暂无球员数据</td></tr>'}</tbody></table></section>
<section class="section"><h2>比赛事件摘要 <span class="pill">候选结果支持人工复核</span></h2><table><thead><tr><th>时间</th><th>队伍</th><th>球员</th><th>距离</th><th>状态</th></tr></thead><tbody>{pass_html or '<tr><td colspan="5">暂无传球事件</td></tr>'}</tbody></table></section>
<section class="section"><h2>数据可信度</h2><div class="note"><div><b>动态标定</b><span>{'已通过' if cal.get('status')=='ready' else '待检查'} · 有效覆盖 {coverage_text}</span></div><div><b>球员身份</b><span>系统区分技术 ID 与人工确认身份；已确认 {sum(1 for v in identity_map.values() if v.get('name'))} 个技术 ID</span></div><div><b>传球人工复核</b><span>已复核 {pass_review_state.get('labeled',0)}/{pass_review_state.get('total',0)} 条{(' · 一致率 '+format(float(pass_review_state['agreement_rate'])*100,'.1f')+'%') if pass_review_state.get('agreement_rate') is not None else ''}</span></div></div></section>
<footer><span>项目 ID：{html.escape(project.get('id',''))}</span><span>视频：{_f(video.get('duration_seconds'))/60:.1f} 分钟 · {video.get('width','—')}×{video.get('height','—')} · {_f(video.get('fps')):.2f} FPS</span><span>由赛场洞察系统自动生成</span></footer>
</body></html>'''
    html_path = output_root / "match_report.html"
    html_path.write_text(doc, encoding="utf-8")
    return html_path, None


def build_player_report(project: dict[str, Any], output_root: Path, player_index: int, *, make_pdf: bool = False) -> tuple[Path, Path | None]:
    """Generate the formal one-player card report used by the review workflow."""
    from app.services.results import player_events, players
    from app.services.reviews import load_player_report_annotation

    rows = players(project)
    if player_index < 0 or player_index >= len(rows):
        raise IndexError(player_index)
    player = rows[player_index]
    gids = [int(x) for x in player.get("global_ids") or []]
    assessment = player.get("assessment") or {}
    scores = assessment.get("scores") or {}
    # The established player-card PDF uses these eight compact labels. Keep
    # that contract while sourcing every value from the corresponding manual
    # UI dimension (never infer a missing score from running data).
    score_labels = {
        "physical": "体", "control": "技", "passing": "战", "endurance": "心",
        "running": "智", "defense": "观", "shooting": "决", "speed": "位",
    }
    radar = {score_labels[key]: scores.get(key) for key in score_labels}
    confirmed = assessment.get("status") == "confirmed" and all(value is not None for value in radar.values())
    match = project.get("match") or {}
    event_rows = player_events(project, gids)
    semantic_counts = {
        "shield": sum(row.get("type") == "shield" for row in event_rows),
        "counterpress": sum(row.get("type") == "counterpress" for row in event_rows),
        "goal": sum(row.get("type") == "goal" for row in event_rows),
    }
    if any(semantic_counts.values()):
        key_metrics = [
            {"key": "shielding_under_pressure", "value": semantic_counts["shield"], "display": f"{semantic_counts['shield']} 次", "label": "对抗护球", "quote": "在对手近身施压下仍保持稳定球权的候选。"},
            {"key": "counterpress_recovery", "value": semantic_counts["counterpress"], "display": f"×{semantic_counts['counterpress']}", "label": "丢球反抢", "quote": "丢球后 5 秒内由本队重新获得稳定球权。"},
            {"key": "goal_candidate", "value": semantic_counts["goal"], "display": str(semantic_counts["goal"]), "label": "进球", "quote": "射门后进入球门区域的候选，最终结论以人工复核为准。"},
        ]
    else:
        key_metrics = [
            {"key": "total_distance_m", "value": player.get("total_distance_m"), "label": "本场跑动距离"},
            {"key": "max_speed_mps", "value": player.get("max_speed_mps"), "label": "最高速度"},
            {"key": "sprint_count", "value": player.get("sprint_count"), "label": "冲刺次数"},
        ]
    note = str(assessment.get("note") or "").strip()
    annotation = load_player_report_annotation(project, gids[0])["fields"] if gids else {}
    recommended_positions = []
    for index in range(1, 4):
        value = annotation.get(f"position_{index}", "").strip()
        if not value:
            continue
        fit_value = annotation.get(f"position_{index}_fit", "").strip()
        recommended_positions.append({
            "rank": index, "position": value,
            "fit": _f(fit_value, None) if fit_value else None,
            "description": annotation.get(f"position_{index}_description") or "人工标注的建议位置",
            "verdict": annotation.get(f"position_{index}_verdict") or "待持续观察",
        })
    payload = {
        "schema_version": "nati-assessment-input-v1",
        "data_status": "final" if confirmed else "partial",
        "player": {
            "player_id": player.get("person_key") or player.get("player_id") or f"ID-{gids[0] if gids else 'unknown'}",
            "display_name": player.get("player_id") or "待确认球员",
            "team": player.get("team") or "待确认队伍",
            "jersey_number": _i(player.get("jersey_number"), 0),
            "age_group": match.get("age_group") or "待补充",
            "position": annotation.get("position") or "待人工标注",
            "preferred_foot": annotation.get("preferred_foot") or "待人工标注",
            "club": annotation.get("club") or player.get("team") or "待补充俱乐部",
            "registry_status": "人工已确认" if player.get("identity_status") == "human_confirmed" else "身份待确认",
        },
        "match": {
            "report_id": f"FI-{project.get('id','')}-P{player_index + 1:03d}",
            "assessment_date": match.get("match_date") or "待评估",
            "tournament": match.get("competition") or project.get("name") or "比赛分析",
            "season": match.get("match_date", "")[:4] if match.get("match_date") else "待补充",
            "match_accolade": f"本场单球员评估 · 技术 ID {', '.join(map(str, gids))}",
        },
        "headline": {
            "nickname": annotation.get("nickname") or ("评估完成" if confirmed else "待人工完善"),
            "quote": annotation.get("quote") or note or "已汇总该球员关联技术 ID 的客观数据与事件；语义结论等待人工标注。",
        },
        "key_metrics": key_metrics,
        "radar": {"dimensions": radar, "scale": [0, 100]},
        "overall": {
            "ca_score": round(sum(float(v) for v in radar.values()) / 8, 1) if confirmed else None,
            "potential_grade": annotation.get("potential_grade") or None,
            "potential_direction": annotation.get("potential_direction") or None,
        },
        "analysis": {
            "strengths": [note] if note else [],
            "improvements": [],
            "strengths_summary": annotation.get("strengths_summary") or note or "等待人工结合球员总视频和候选事件填写正式观察。",
            "improvements_summary": annotation.get("improvements_summary") or "待人工标注后生成训练建议。",
        },
        "style_archetype": {
            "triangle": {
                "tactical_literacy": _f(annotation.get("tactical_literacy"), None) if annotation.get("tactical_literacy") else None,
                "physical_competition": _f(annotation.get("physical_competition"), None) if annotation.get("physical_competition") else None,
                "talent": _f(annotation.get("talent"), None) if annotation.get("talent") else None,
            },
            "reference_player": annotation.get("reference_player") or None,
            "style_tag": annotation.get("style_tag") or None,
            "narrative": annotation.get("style_narrative") or None,
            "similarities": [line.strip() for line in annotation.get("similarities", "").splitlines() if line.strip()],
            "differences": [line.strip() for line in annotation.get("differences", "").splitlines() if line.strip()],
            "next_target": annotation.get("next_target") or None,
        },
        "position_recommendations": recommended_positions,
        "messages": {
            "to_player": annotation.get("to_player") or note or "本报告的客观数据已生成，技术与战术结论等待人工评估。",
            "to_family_and_coach": annotation.get("to_family_and_coach") or "多个技术 ID 已按人工身份映射合并；报告保留原始 ID 作为可追溯证据。",
        },
        "evidence": {"event_count": len(event_rows), "highlight_count": len(event_rows), "qr_caption": "球员证据\n见系统工作台"},
    }
    engine_root = Path(__file__).resolve().parents[2] / "engine" / "match_analysis"
    if str(engine_root) not in sys.path:
        sys.path.insert(0, str(engine_root))
    from render_nati_standard_report import build_html, normalize_payload, render_pdf  # type: ignore

    normalized = normalize_payload(payload)
    html_text = build_html(normalized)
    report_root = output_root / "player_reports"
    report_root.mkdir(parents=True, exist_ok=True)
    stem = f"player_{player_index + 1:03d}_ids_{'-'.join(map(str, gids)) or 'unknown'}"
    html_path = report_root / f"{stem}.html"
    json_path = report_root / f"{stem}.json"
    html_path.write_text(html_text, encoding="utf-8")
    json_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    pdf_path = report_root / f"{stem}.pdf"
    if make_pdf:
        render_pdf(html_text, pdf_path, keep_html=html_path)
    return html_path, pdf_path if pdf_path.is_file() else None
