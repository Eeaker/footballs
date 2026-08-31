(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s),
    $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const state = {
    projects: [],
    project: null,
    projectFilter: "all",
    page: "projects",
    pollTimer: null,
    previewPollTimer: null,
    results: {},
    leaderMode: "distance",
    playerIndex: 0,
    calib: {
      imagePoints: [],
      worldPoints: [],
      validations: [],
      validationDraft: [],
      clickMode: null,
      sourceW: 0,
      sourceH: 0,
      imported: null,
      importedProjectId: null,
      dynamicVisual: null,
      visualToken: 0,
      playing: false,
      timer: null,
      timelineProjectId: null,
      timelineSignature: null,
      frameToken: 0,
      scrubbing: false,
      scrubRaf: null,
      pendingSeekFrame: null,
      visualTimer: null,
      lastVisualFrame: -1,
      dragFrameInFlight: false,
      dragPendingFrame: null,
      dragPointerId: null,
    },
    replay: {
      data: null,
      index: 0,
      playing: false,
      timer: null,
      speed: 1,
      mode: "live",
      homographies: new Map(),
      visualToken: 0,
      pitchStartedAt: 0,
      pitchStartTime: 0,
      pitchTime: 0,
      compilationManifest: null,
      windowSize: 1800,
      windowCache: new Map(),
      windowLoading: null,
      loadingPromise: null,
      compilationFullscreenAbort: null,
    },
    system: null,
  };
  const titles = {
    projects: "项目中心",
    setup: "新建 / 配置分析",
    progress: "分析进度",
    results: "结果中心",
    system: "系统状态",
  };
  const teamColor = {
    team_0: "#4b8cff",
    team_1: "#ff9b43",
    team_2: "#9a70ff",
    unassigned: "#8fa2b8",
  };
  const settingDefs = {
    basic: [
      [
        "expected_players",
        "场上人数先验",
        "number",
        1,
        40,
        1,
        "用于场景先验，不强制把技术 ID 压到这个数量",
      ],
      [
        "team_clusters",
        "队伍分组数",
        "number",
        2,
        4,
        1,
        "通常两队 + 裁判/其他 = 3",
      ],
      [
        "confidence",
        "检测置信度",
        "number",
        0.05,
        0.9,
        0.01,
        "正式分析建议 0.20–0.30",
      ],
      [
        "imgsz",
        "分析分辨率",
        "select",
        null,
        null,
        null,
        "越高越慢，但远景小球员更有利",
        [736, 960, 1280, 1536],
      ],
      [
        "event_count",
        "重点事件数量",
        "number",
        5,
        100,
        1,
        "用于事件摘要和高光候选",
      ],
      [
        "min_pass_displacement_m",
        "最低传球位移(m)",
        "number",
        0.2,
        5,
        0.1,
        "低于该距离的同队转换不作为主动传球候选",
      ],
    ],
    advanced: [
      [
        "device",
        "计算设备",
        "text",
        null,
        null,
        null,
        "GPU 通常为 0；CPU 填 cpu",
      ],
      [
        "weights_path",
        "分析模型路径",
        "text",
        null,
        null,
        null,
        "默认 models/yolov8x.pt",
      ],
      [
        "min_track_frames",
        "最短轨迹帧数",
        "number",
        2,
        300,
        1,
        "过滤极短碎片轨迹",
      ],
      [
        "min_presence_ratio",
        "最小出场占比",
        "number",
        0,
        0.5,
        0.001,
        "技术轨迹存在时间的最低占比",
      ],
      ["min_turf_score", "草地置信阈值", "number", 0, 1, 0.01, "场内过滤参数"],
      [
        "min_track_turf_ratio",
        "轨迹草地占比",
        "number",
        0,
        1,
        0.01,
        "整条轨迹的草地证据",
      ],
      [
        "min_foot_y_ratio",
        "脚点高度阈值",
        "number",
        0,
        1,
        0.01,
        "过滤明显场外区域",
      ],
      [
        "event_percentile",
        "事件分位阈值",
        "number",
        50,
        99.9,
        0.5,
        "关键动作候选阈值",
      ],
      [
        "event_min_gap",
        "事件最小间隔(s)",
        "number",
        0.2,
        20,
        0.1,
        "避免相邻帧重复报事件",
      ],
      ["pre_sec", "高光前置(s)", "number", 0, 10, 0.5, "事件前保留时长"],
      ["post_sec", "高光后置(s)", "number", 0, 10, 0.5, "事件后保留时长"],
      [
        "team_samples_per_id",
        "队色采样数",
        "number",
        3,
        60,
        1,
        "每个技术 ID 的队伍证据采样",
      ],
      [
        "ocr_candidates_per_id",
        "号码候选帧数",
        "number",
        4,
        100,
        1,
        "多帧号码证据上限",
      ],
      [
        "identity_audit_enabled",
        "身份质量审计",
        "choice",
        null,
        null,
        null,
        "检查技术 ID 是否存在跨外观模式污染，不自动改 ID",
        [
          [true, "开启"],
          [false, "关闭"],
        ],
      ],
      [
        "identity_audit_sample_stride",
        "身份审计采样间隔",
        "number",
        5,
        120,
        1,
        "用于学习队服外观模式；值越小越慢",
      ],
      [
        "dynamic_sample_step",
        "动态标定采样间隔",
        "number",
        1,
        30,
        1,
        "旋转越快建议越小",
      ],
      [
        "dynamic_max_gap",
        "动态标定最大插值缺口",
        "number",
        3,
        180,
        1,
        "超出缺口的帧不参与米制分析",
      ],
      [
        "dynamic_min_coverage",
        "动态标定最低有效覆盖",
        "number",
        0.5,
        1,
        0.01,
        "低于该覆盖率阻止正式分析",
      ],
      [
        "calibration_tolerance_m",
        "尺度验证误差阈值(m)",
        "number",
        0.05,
        3,
        0.05,
        "每个标定锚点独立验证",
      ],
      [
        "focus_clip_limit",
        "球员高光上限",
        "number",
        2,
        40,
        1,
        "报告阶段自动生成的目标片段数量",
      ],
    ],
  };
  const metaDefs = [
    ["competition", "赛事/比赛名称"],
    ["match_date", "比赛日期"],
    ["venue", "场地"],
    ["age_group", "年龄组"],
    ["home_team", "主队"],
    ["away_team", "客队"],
  ];
  const assessmentDefs = [
    ["speed", "速度"],
    ["endurance", "耐力"],
    ["running", "跑动"],
    ["passing", "传球"],
    ["control", "控球"],
    ["shooting", "射门"],
    ["defense", "防守"],
    ["physical", "对抗"],
  ];

  async function api(url, opt = {}) {
    const res = await fetch(url, opt);
    const type = res.headers.get("content-type") || "";
    let data;
    if (type.includes("application/json")) data = await res.json();
    else data = await res.text();
    if (!res.ok) {
      const msg = (data && data.detail) || data || `HTTP ${res.status}`;
      throw new Error(msg);
    }
    return data;
  }
  function toast(msg, type = "") {
    const el = $("#toast");
    el.textContent = msg;
    el.className = `toast ${type} show`;
    clearTimeout(el._t);
    el._t = setTimeout(() => (el.className = "toast"), 3200);
  }
  const fmtBytes = (n) => {
    n = Number(n) || 0;
    if (n < 1024) return `${n} B`;
    if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
    return `${(n / 1024 ** 3).toFixed(1)} GB`;
  };
  const fmtTime = (s) => {
    s = Math.max(0, Number(s) || 0);
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  };
  const fmtDate = (s) => {
    if (!s) return "—";
    try {
      return new Date(s).toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return s;
    }
  };
  const esc = (s) =>
    String(s ?? "").replace(
      /[&<>'"]/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          "'": "&#39;",
          '"': "&quot;",
        })[c],
    );
  const eventLabels = {
    shielding_under_pressure: "对抗护球",
    shield: "对抗护球",
    counterpress_recovery: "丢球反抢",
    counterpress: "丢球反抢",
    goal_candidate: "进球候选",
    goal: "进球候选",
    pass: "传球候选",
    turnover: "球权转换",
    key: "关键动作",
    possession_transition: "球权转换",
  };
  const eventLabel = (value) => eventLabels[value] || String(value || "候选事件");
  const identityLabel = (value) =>
    ({
      human_confirmed: "人工已确认",
      confirmed: "已确认",
      candidate: "待确认",
      pending: "待确认",
      unassigned: "未分组",
    })[value] || String(value || "待确认");
  const videoRevision = (p) => {
    const v = p?.video;
    return v
      ? `${p.id}:${v.size_bytes || 0}:${v.frame_count || 0}:${v.fps || 0}:${v.preview_status || "source"}:${v.preview_url || ""}`
      : "";
  };
  function syncVideoSource(video, p, preview = true) {
    const key = `${videoRevision(p)}:${preview ? "preview" : "source"}`;
    if (!video || !key) return;
    if (video.dataset.sourceKey === key) return;
    video.dataset.sourceKey = key;
    const endpoint = preview
      ? p.video?.preview_url || `/api/projects/${p.id}/preview-video`
      : `/api/projects/${p.id}/source-video`;
    video.src = `${endpoint}?v=${encodeURIComponent(key)}`;
    video.load();
  }
  function schedulePreviewStatusPoll(p) {
    clearTimeout(state.previewPollTimer);
    state.previewPollTimer = null;
    if (p?.video?.preview_status !== "building") return;
    const projectId = p.id;
    state.previewPollTimer = setTimeout(async () => {
      try {
        const fresh = await api(`/api/projects/${projectId}`);
        if (state.project?.id !== projectId) return;
        state.project = fresh;
        renderSetup();
      } catch {}
    }, 3000);
  }
  function clearVideoSource(video) {
    if (!video || !video.getAttribute("src")) return;
    video.pause();
    video.removeAttribute("src");
    delete video.dataset.sourceKey;
    video.load();
  }
  const calibrationFrameUrl = (frame) =>
    `/api/projects/${state.project.id}/frame?frame_index=${frame}&v=${encodeURIComponent(videoRevision(state.project))}`;
  function statusLabel(p) {
    const s = p?.pipeline?.state || p?.status;
    return (
      {
        complete: "分析完成",
        running: "分析中",
        failed: "处理失败",
        cancelled: "已取消",
        interrupted: "已中断",
        queued: "排队中",
        idle: "待配置",
        draft: "待配置",
        configured: "待标定",
      }[s] || "待配置"
    );
  }

  async function loadProjects(prefer) {
    state.projects = await api("/api/projects");
    const sel = $("#projectSelect");
    sel.innerHTML = state.projects
      .map(
        (p) =>
          `<option value="${esc(p.id)}">${esc(p.name)}${p.kind === "demo" ? " · 示例" : ""}</option>`,
      )
      .join("");
    const target = prefer || state.project?.id || state.projects[0]?.id;
    if (target && state.projects.some((p) => p.id === target)) {
      sel.value = target;
      await loadProject(target, false);
    } else state.project = null;
    renderProjectCards();
  }
  async function loadProject(id, refreshResults = false) {
    const previousId = state.project?.id;
    state.project = await api(`/api/projects/${id}`);
    if (previousId && previousId !== id) {
      setCalibPlaying(false);
      setReplayPlaying(false);
      state.calib.imported = null;
      state.calib.importedProjectId = null;
      state.calib.dynamicVisual = null;
      state.calib.timelineProjectId = null;
      state.calib.timelineSignature = null;
      state.replay.data = null;
      state.replay.windowCache = new Map();
      state.replay.loadingPromise = null;
      state.replay.windowLoading = null;
      state.replay.homographies = new Map();
      clearAnchorDraft();
    }
    $("#projectSelect").value = id;
    renderSetup();
    renderProgress();
    managePolling();
    if (refreshResults || state.page === "results") await loadResults();
  }
  function renderProjectCards() {
    const root = $("#projectCards");
    let rows = state.projects;
    if (state.projectFilter === "analysis")
      rows = rows.filter((p) => p.kind !== "demo");
    if (state.projectFilter === "complete")
      rows = rows.filter(
        (p) => p.status === "complete" || p.pipeline?.state === "complete",
      );
    root.innerHTML =
      rows
        .map((p) => {
          const selected = p.id === state.project?.id;
          const v = p.video || {};
          const st = p.pipeline?.state || p.status;
          const cls = st === "complete" ? "ok" : st === "failed" ? "bad" : "";
          return `<article class="project-card ${selected ? "selected" : ""}" data-project-id="${esc(p.id)}"><div class="thumb"><div class="pitch-mini"></div><span class="project-status ${cls}">${esc(statusLabel(p))}</span></div><h4>${esc(p.name)}</h4><p>${esc((p.match || {}).home_team || "主队")} vs ${esc((p.match || {}).away_team || "客队")} · ${p.kind === "demo" ? "系统示例" : "正式项目"}</p><div class="project-stats"><span>◷ ${v.duration_seconds ? `${(v.duration_seconds / 60).toFixed(1)}min` : "未上传"}</span><span>▣ ${v.width ? `${v.width}×${v.height}` : "—"}</span><span>↻ ${fmtDate(p.updated_at)}</span></div>${p.kind !== "demo" ? `<div class="project-actions"><button title="删除项目" data-delete-project="${esc(p.id)}">×</button></div>` : ""}</article>`;
        })
        .join("") ||
      '<div class="empty-state"><h3>还没有比赛项目</h3><p>点击右上角新建第一场正式分析。</p></div>';
    $$("[data-project-id]", root).forEach((card) =>
      card.addEventListener("click", async (e) => {
        if (e.target.closest("[data-delete-project]")) return;
        await loadProject(card.dataset.projectId);
        switchPage(state.project?.status === "complete" ? "results" : "setup");
      }),
    );
    $$("[data-delete-project]", root).forEach((btn) =>
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const p = state.projects.find(
          (x) => x.id === btn.dataset.deleteProject,
        );
        if (
          !confirm(
            `删除“${p?.name || "该项目"}”？项目视频、标定和结果都会删除。`,
          )
        )
          return;
        try {
          await api(`/api/projects/${btn.dataset.deleteProject}`, {
            method: "DELETE",
          });
          toast("项目已删除", "success");
          state.project = null;
          await loadProjects();
        } catch (err) {
          toast(err.message, "error");
        }
      }),
    );
  }

  function switchPage(page) {
    state.page = page;
    $$(".page").forEach((x) =>
      x.classList.toggle("active", x.id === `page-${page}`),
    );
    $$(".nav-item[data-page]").forEach((x) =>
      x.classList.toggle("active", x.dataset.page === page),
    );
    $("#pageTitle").textContent = titles[page] || "";
    if (page === "projects") renderProjectCards();
    if (page === "setup") {
      renderSetup();
      requestAnimationFrame(drawCalibOverlay);
    }
    if (page === "progress") renderProgress();
    if (page === "results")
      loadResults().catch((e) => toast(e.message, "error"));
    if (page === "system") loadSystem().catch((e) => toast(e.message, "error"));
  }

  function renderMetaForm() {
    const p = state.project;
    const root = $("#matchMetaForm");
    if (!p) return;
    root.innerHTML = metaDefs
      .map(
        ([key, label]) =>
          `<label>${label}<input data-meta="${key}" value="${esc((p.match || {})[key] || "")}" /></label>`,
      )
      .join("");
  }
  function renderSettings() {
    const s = state.project?.settings;
    if (!s) return;
    const render = (defs, root) => {
      root.innerHTML = defs
        .map((d) => {
          const [key, label, type, min, max, step, hint, opts] = d;
          let input;
          if (type === "select")
            input = `<select data-setting="${key}">${opts.map((v) => `<option value="${v}" ${String(s[key]) === String(v) ? "selected" : ""}>${v}px</option>`).join("")}</select>`;
          else if (type === "choice")
            input = `<select data-setting="${key}" data-value-type="choice">${opts.map((v) => `<option value="${v[0]}" ${String(s[key]) === String(v[0]) ? "selected" : ""}>${esc(v[1])}</option>`).join("")}</select>`;
          else
            input = `<input data-setting="${key}" type="${type}" value="${esc(s[key])}" ${min != null ? `min="${min}"` : ""} ${max != null ? `max="${max}"` : ""} ${step != null ? `step="${step}"` : ""}/>`;
          return `<div class="settings-row"><div class="setting-head"><b>${label}</b><span>${hint}</span></div>${input}</div>`;
        })
        .join("");
    };
    render(settingDefs.basic, $("#settingsForm"));
    render(settingDefs.advanced, $("#advancedSettings"));
    $("#fieldLength").value = s.field_length_m;
    $("#fieldWidth").value = s.field_width_m;
    $("#calibTolerance").value = s.calibration_tolerance_m;
    $("#coverageRequirement").textContent =
      `≥ ${(Number(s.dynamic_min_coverage || 0.8) * 100).toFixed(0)}%`;
  }
  function renderSetup() {
    const p = state.project;
    const formal = p && p.kind !== "demo";
    $("#setupEmpty").classList.toggle("hidden", !!formal);
    $("#setupContent").classList.toggle("hidden", !formal);
    if (!formal) return;
    const v = p.video;
    $("#videoState").textContent = v ? "已上传" : "未上传";
    $("#videoState").className = `state-chip ${v ? "ok" : ""}`;
    $("#videoMeta").innerHTML = v
      ? [
          ["时长", `${(v.duration_seconds / 60).toFixed(1)} min`],
          ["FPS", Number(v.fps).toFixed(2)],
          ["分辨率", `${v.width}×${v.height}`],
          ["文件", fmtBytes(v.size_bytes)],
          [
            "视角",
            v.health?.motion_type === "pan_rotate"
              ? "固定机位 · 旋转视角"
              : v.health?.motion_type === "fixed"
                ? "固定视角"
                : "动态标定模式",
          ],
        ]
          .map(
            (x) =>
              `<div class="meta-item"><span>${x[0]}</span><b>${x[1]}</b></div>`,
          )
          .join("")
      : "";
    const preview = $("#videoPreview"),
      scrubVideo = $("#calibScrubVideo");
    if (v) {
      syncVideoSource(preview, p);
      syncVideoSource(scrubVideo, p);
      preview.classList.remove("hidden");
      schedulePreviewStatusPoll(p);
    } else {
      clearVideoSource(preview);
      clearVideoSource(scrubVideo);
      preview.classList.add("hidden");
      schedulePreviewStatusPoll(null);
    }
    renderMetaForm();
    renderSettings();
    $("#exportSettingsBtn").href = `/api/projects/${p.id}/settings/export`;
    const roster = p.roster || {};
    $("#rosterState").textContent =
      roster.status === "ready" ? `${roster.count} 人` : "未上传";
    $("#rosterState").className =
      `state-chip ${roster.status === "ready" ? "ok" : "neutral"}`;
    renderCalibration();
    renderCalibTimeline();
    refreshPreflight().catch(() => {});
  }
  function renderUploadedCalibration(c) {
    const root = $("#uploadedCalibrationSummary"),
      v = c.validation || {},
      show = c.status === "ready" && c.source === "uploaded_dynamic";
    root.classList.toggle("hidden", !show);
    if (!show) return;
    const ratio = v.accepted_ratio,
      accepted = Number(v.accepted_frames) || 0,
      total = Number(v.total_frames) || 0,
      fps = Number(v.fps) || Number(state.project?.video?.fps) || 0;
    const start = v.valid_start_frame,
      end = v.valid_end_frame,
      frameRange =
        start != null && end != null
          ? `${Number(start).toLocaleString("zh-CN")}–${Number(end).toLocaleString("zh-CN")}`
          : "全片";
    const durationRange =
      start != null && end != null && fps
        ? `${fmtTime(Number(start) / fps)}–${fmtTime(Number(end) / fps)}`
        : "全片时段";
    const field =
      v.field_length_m != null && v.field_width_m != null
        ? `${Number(v.field_length_m).toFixed(1)} × ${Number(v.field_width_m).toFixed(1)} m`
        : "未提供";
    $("#uploadedCalibrationName").textContent =
      c.source_filename || "dynamic_calibration.json";
    $("#uploadedCalibrationMeta").textContent =
      `${fmtDate(c.uploaded_at)} · 与当前视频匹配`;
    const facts = [
      [
        "有效覆盖",
        ratio != null ? `${(Number(ratio) * 100).toFixed(1)}%` : "已验证",
      ],
      [
        "有效帧",
        accepted && total
          ? `${accepted.toLocaleString("zh-CN")} / ${total.toLocaleString("zh-CN")}`
          : "已验证",
      ],
      ["帧范围", frameRange],
      ["对应时段", durationRange],
      [
        "画面规格",
        v.frame_width && v.frame_height
          ? `${v.frame_width}×${v.frame_height} · ${fps.toFixed(2)} FPS`
          : "与视频一致",
      ],
      ["球场范围", field],
    ];
    $("#uploadedCalibrationFacts").innerHTML = facts
      .map((x) => `<div><span>${x[0]}</span><b>${esc(x[1])}</b></div>`)
      .join("");
    const anchorFrames = (v.anchor_proc_indices || [])
      .map(Number)
      .filter(Number.isFinite);
    const anchorText = v.anchor_count
      ? `${v.anchor_count} 个${anchorFrames.length ? ` · 帧 ${anchorFrames.map((x) => x.toLocaleString("zh-CN")).join("、")}` : ""}`
      : v.reference_frame != null
        ? `参考帧 ${Number(v.reference_frame).toLocaleString("zh-CN")}`
        : "未声明";
    const sampling = v.sample_step_frames
      ? `每 ${v.sample_step_frames} 帧采样${v.max_interpolation_gap_frames ? ` · 最大插值缺口 ${v.max_interpolation_gap_frames} 帧` : ""}`
      : "逐帧配置";
    const validation =
      v.validation_tolerance_m != null
        ? `阈值 ${Number(v.validation_tolerance_m).toFixed(2)} m${v.validation_max_error_m != null ? ` · 最大误差 ${Number(v.validation_max_error_m).toFixed(3)} m` : ""}${v.validation_segment_count ? ` · ${v.validation_segment_count} 条独立线段` : ""}`
        : "独立验证已通过";
    const details = [
      [
        "格式",
        `Schema v${v.schema_version ?? "—"} · stride ${v.vid_stride ?? 1}`,
      ],
      ["视角锚点", anchorText],
      ["动态注册", sampling],
      ["尺度验证", validation],
    ];
    if (v.registration_method)
      details.push(["注册方法", v.registration_method]);
    $("#uploadedCalibrationDetails").innerHTML = details
      .map((x) => `<div><span>${x[0]}</span><b>${esc(x[1])}</b></div>`)
      .join("");
  }
  async function loadUploadedCalibrationVisual(c) {
    const projectId = state.project?.id;
    if (!projectId || c.status !== "ready" || c.source !== "uploaded_dynamic")
      return;
    if (state.calib.importedProjectId === projectId && state.calib.imported) {
      updateCalibLists();
      requestAnimationFrame(drawCalibOverlay);
      return;
    }
    const data = await api(
      `/api/projects/${projectId}/calibration/visualization`,
    );
    if (state.project?.id !== projectId) return;
    state.calib.imported = { ...data, projectId };
    state.calib.importedProjectId = projectId;
    c.validation = { ...(c.validation || {}), ...(data.summary || {}) };
    renderUploadedCalibration(c);
    const bounds = data.field_bounds_m || {},
      length = Number(bounds.x_max) - Number(bounds.x_min),
      width = Number(bounds.y_max) - Number(bounds.y_min);
    if (Number.isFinite(length) && length > 0) $("#fieldLength").value = length;
    if (Number.isFinite(width) && width > 0) $("#fieldWidth").value = width;
    $("#calibFrame").value = data.frame_index;
    updateCalibLists();
    loadCalibrationFrame(Number(data.frame_index), true);
  }
  function preciseTime(seconds) {
    seconds = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(seconds / 60),
      secs = Math.floor(seconds % 60),
      hundredths = Math.floor((seconds % 1) * 100);
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(hundredths).padStart(2, "0")}`;
  }
  function renderCalibTimeline() {
    const v = state.project?.video;
    if (!v) return;
    const total = Math.max(1, Number(v.frame_count) || 1),
      fps = Number(v.fps) || 30,
      slider = $("#calibTimelineSlider"),
      signature = videoRevision(state.project);
    slider.max = total - 1;
    const current = Math.max(
      0,
      Math.min(Number($("#calibFrame").value) || 0, total - 1),
    );
    slider.value = current;
    $("#calibTimelineDuration").textContent =
      `/ ${preciseTime((total - 1) / fps)}`;
    if (state.calib.timelineSignature !== signature) {
      const ticks = 9,
        thumbs = 12;
      $("#calibTimelineRuler").innerHTML = Array.from(
        { length: ticks },
        (_, i) => {
          const frame = Math.round((i / (ticks - 1)) * (total - 1));
          return `<span style="left:${(i / (ticks - 1)) * 100}%">${fmtTime(frame / fps)}</span>`;
        },
      ).join("");
      $("#calibTimelineStrip").innerHTML = Array.from(
        { length: thumbs },
        (_, i) => {
          const frame = Math.round((i / (thumbs - 1)) * (total - 1));
          return `<button type="button" data-timeline-frame="${frame}" title="帧 ${frame.toLocaleString("zh-CN")} · ${preciseTime(frame / fps)}"><img loading="lazy" decoding="async" src="${calibrationFrameUrl(frame)}" alt="${fmtTime(frame / fps)}"><span>${fmtTime(frame / fps)}</span></button>`;
        },
      ).join("");
      state.calib.timelineSignature = signature;
      state.calib.timelineProjectId = state.project.id;
    }
    updateCalibTimeline(current);
  }
  function updateCalibTimeline(frame) {
    const v = state.project?.video;
    if (!v) return;
    const total = Math.max(1, Number(v.frame_count) || 1),
      fps = Number(v.fps) || 30,
      fi = Math.max(0, Math.min(Number(frame) || 0, total - 1));
    $("#calibTimelineSlider").value = fi;
    $("#calibTimelineCurrent").textContent = preciseTime(fi / fps);
    $("#calibTimelinePlayhead").style.left =
      `${(fi / Math.max(1, total - 1)) * 100}%`;
    $$("[data-timeline-frame]").forEach((x) =>
      x.classList.toggle(
        "active",
        Math.abs(Number(x.dataset.timelineFrame) - fi) <=
          Math.max(1, total / 20),
      ),
    );
  }
  async function loadDynamicFrameVisual(frame) {
    const c = state.project?.calibration || {},
      badge = $("#dynamicCalibBadge");
    if (c.status !== "ready" || !c.path) {
      state.calib.dynamicVisual = null;
      badge.classList.add("hidden");
      drawCalibOverlay();
      return;
    }
    const token = ++state.calib.visualToken;
    badge.classList.remove("hidden");
    badge.classList.add("loading");
    badge.querySelector("b").textContent = "正在计算画面映射";
    try {
      const visual = await api(
        `/api/projects/${state.project.id}/calibration/frame-visualization?frame_index=${frame}`,
      );
      if (token !== state.calib.visualToken) return;
      state.calib.dynamicVisual = visual;
      badge.classList.remove("loading");
      badge.classList.toggle("bad", !visual.accepted);
      badge.querySelector("b").textContent = visual.accepted
        ? `帧 #${Number(visual.frame_index).toLocaleString("zh-CN")} · 映射有效`
        : `帧 #${Number(visual.frame_index).toLocaleString("zh-CN")} · 映射无效`;
      drawCalibOverlay();
    } catch (e) {
      if (token !== state.calib.visualToken) return;
      state.calib.dynamicVisual = null;
      badge.classList.remove("loading");
      badge.classList.add("bad");
      badge.querySelector("b").textContent = "该帧标定读取失败";
      drawCalibOverlay();
    }
  }
  function setCalibFrameUi(frame, label = "视频帧") {
    const v = state.project?.video;
    if (!v) return 0;
    const total = Math.max(1, Number(v.frame_count) || 1),
      fps = Number(v.fps) || 30,
      fi = Math.max(0, Math.min(Math.round(Number(frame) || 0), total - 1));
    $("#calibFrame").value = fi;
    updateCalibTimeline(fi);
    $("#calibFrameBadge").textContent =
      `${label} #${fi.toLocaleString("zh-CN")} · ${preciseTime(fi / fps)}`;
    $("#calibFrameBadge").classList.remove("hidden");
    return fi;
  }
  function scheduleDynamicFrameVisual(frame, delay = 110) {
    clearTimeout(state.calib.visualTimer);
    state.calib.visualTimer = setTimeout(
      () => loadDynamicFrameVisual(frame),
      delay,
    );
  }
  function seekCalibScrubVideo() {
    state.calib.scrubRaf = null;
    const video = $("#calibScrubVideo"),
      frame = state.calib.pendingSeekFrame,
      v = state.project?.video;
    if (frame == null || !v || video.readyState < 1) return;
    state.calib.pendingSeekFrame = null;
    const target = Math.min(
      Number(video.duration) || Infinity,
      frame / (Number(v.fps) || 30),
    );
    if (
      Number.isFinite(target) &&
      Math.abs((video.currentTime || 0) - target) > 0.001
    )
      video.currentTime = target;
  }
  function showNearestTimelineFrame(frame) {
    const items = $$("[data-timeline-frame]");
    if (!items.length) return;
    const nearest = items.reduce((best, item) =>
        Math.abs(Number(item.dataset.timelineFrame) - frame) <
        Math.abs(Number(best.dataset.timelineFrame) - frame)
          ? item
          : best,
      ),
      thumb = nearest.querySelector("img"),
      img = $("#calibImage");
    if (thumb?.complete && thumb.naturalWidth) {
      img.src = thumb.currentSrc || thumb.src;
      state.calib.sourceW = state.project.video.width;
      state.calib.sourceH = state.project.video.height;
    }
  }
  function pumpCalibrationDragFrame() {
    if (
      state.calib.dragFrameInFlight ||
      state.calib.dragPendingFrame == null ||
      !state.project?.video
    )
      return;
    const fi = state.calib.dragPendingFrame,
      projectId = state.project.id,
      img = $("#calibImage"),
      video = $("#calibScrubVideo");
    state.calib.dragPendingFrame = null;
    state.calib.dragFrameInFlight = true;
    const loader = new Image();
    loader.decoding = "async";
    const next = () => {
      state.calib.dragFrameInFlight = false;
      if (state.calib.dragPendingFrame != null) pumpCalibrationDragFrame();
    };
    loader.onload = () => {
      const current = Number($("#calibFrame").value),
        fps = Number(state.project?.video?.fps) || 30;
      if (
        state.calib.scrubbing &&
        state.project?.id === projectId &&
        Math.abs(current - fi) <= fps * 2
      ) {
        img.src = loader.src;
        state.calib.sourceW = state.project.video.width;
        state.calib.sourceH = state.project.video.height;
        video.classList.remove("active");
        img.classList.remove("scrub-hidden");
        drawCalibOverlay();
      }
      next();
    };
    loader.onerror = next;
    loader.src = calibrationFrameUrl(fi);
  }
  function previewCalibrationFrame(frame) {
    const v = state.project?.video;
    if (!v) return;
    const fi = setCalibFrameUi(frame),
      video = $("#calibScrubVideo"),
      img = $("#calibImage");
    state.calib.scrubbing = true;
    state.calib.dragPendingFrame = fi;
    video.classList.remove("active");
    img.classList.remove("scrub-hidden");
    showNearestTimelineFrame(fi);
    pumpCalibrationDragFrame();
    state.calib.dynamicVisual = null;
    drawCalibOverlay();
    scheduleDynamicFrameVisual(fi, 220);
  }
  function setCalibPlaying(flag) {
    const video = $("#calibScrubVideo"),
      img = $("#calibImage");
    cancelAnimationFrame(state.calib.timer);
    state.calib.timer = null;
    state.calib.playing = flag;
    $("#calibTimelinePlay").textContent = flag ? "❚❚" : "▶";
    if (!flag) {
      video.pause();
      return;
    }
    const frame = setCalibFrameUi($("#calibFrame").value);
    state.calib.scrubbing = true;
    state.calib.pendingSeekFrame = frame;
    syncVideoSource(video, state.project);
    if (!state.calib.scrubRaf)
      state.calib.scrubRaf = requestAnimationFrame(seekCalibScrubVideo);
    const start = () => {
      if (!state.calib.playing) return;
      video
        .play()
        .then(() => {
          video.classList.add("active");
          img.classList.add("scrub-hidden");
          const tick = () => {
            if (!state.calib.playing || !state.project?.video) return;
            const fps = Number(state.project.video.fps) || 30,
              total = Number(state.project.video.frame_count) || 1,
              fi = Math.min(total - 1, Math.round(video.currentTime * fps));
            setCalibFrameUi(fi);
            if (
              Math.abs(fi - state.calib.lastVisualFrame) >=
              Math.max(1, Math.round(fps / 4))
            ) {
              state.calib.lastVisualFrame = fi;
              scheduleDynamicFrameVisual(fi, 40);
            }
            if (video.ended) {
              setCalibPlaying(false);
              loadCalibrationFrame(fi);
              return;
            }
            state.calib.timer = requestAnimationFrame(tick);
          };
          tick();
        })
        .catch(() => {
          state.calib.playing = false;
          $("#calibTimelinePlay").textContent = "▶";
        });
    };
    if (video.readyState >= 1) start();
    else video.addEventListener("loadedmetadata", start, { once: true });
  }
  function renderCalibration() {
    const c = state.project?.calibration || {};
    let text = "未标定",
      cls = "";
    if (c.status === "ready") {
      text = "动态标定已就绪";
      cls = "ok";
    } else if (c.status === "building") {
      text = "正在生成…";
      cls = "neutral";
    } else if (c.status === "anchors_ready") {
      text = "锚点已保存";
      cls = "neutral";
    } else if (String(c.status).includes("failed")) {
      text = "标定未通过";
      cls = "bad";
    }
    $("#calibrationState").textContent = text;
    $("#calibrationState").className = `state-chip ${cls}`;
    const download = $("#downloadCalibrationBtn");
    if (download) {
      download.href = `/api/projects/${state.project.id}/calibration/download`;
      download.classList.toggle("hidden", c.status !== "ready");
    }
    const anchors = c.anchors || [];
    $("#anchorList").innerHTML = anchors.length
      ? anchors
          .map(
            (a) =>
              `<span class="anchor-chip ${a.passed ? "ok" : ""}">帧 ${a.frame_index} · ${a.passed ? "已验证" : "未通过"} <button data-delete-anchor="${esc(a.id)}">×</button></span>`,
          )
          .join("")
      : '<span class="state-chip neutral">暂无锚点</span>';
    $$("[data-delete-anchor]").forEach((b) =>
      b.addEventListener("click", async () => {
        if (!confirm("删除这个视角锚点？")) return;
        try {
          await api(
            `/api/projects/${state.project.id}/calibration/anchors/${b.dataset.deleteAnchor}`,
            { method: "DELETE" },
          );
          await loadProject(state.project.id);
          toast("锚点已删除", "success");
        } catch (e) {
          toast(e.message, "error");
        }
      }),
    );
    $("#buildDynamicBtn").disabled =
      c.status === "building" || !anchors.some((a) => a.passed);
    if (c.status === "ready" && c.validation) {
      const ratio = c.validation.accepted_ratio;
      $("#referenceResult").className = "calib-result ok";
      $("#referenceResult").innerHTML =
        `<b>动态标定通过</b> · ${anchors.length || c.validation.anchor_count || 1} 个视角锚点 · 有效覆盖 ${ratio != null ? (ratio * 100).toFixed(1) + "%" : "已验证"}<br><span style="color:#7890aa">${esc(c.message || "")}</span>`;
    } else if (c.status === "failed") {
      $("#referenceResult").className = "calib-result bad";
      $("#referenceResult").innerHTML =
        `<b>动态标定未通过</b> · ${esc(c.message || "请增加锚点或调整参数")}`;
    } else $("#referenceResult").classList.add("hidden");
    renderUploadedCalibration(c);
    loadUploadedCalibrationVisual(c).catch((e) =>
      toast(`标定画面读取失败：${e.message}`, "error"),
    );
  }
  async function refreshPreflight() {
    if (!state.project || state.project.kind === "demo") return;
    const data = await api(`/api/projects/${state.project.id}/preflight`);
    $("#preflightGrid").innerHTML = data.checks
      .map(
        (x) =>
          `<div class="preflight-item ${x.ok ? "ok" : "bad"}"><div class="check-dot">${x.ok ? "✓" : "!"}</div><div><b>${esc(x.label)}</b><span>${esc(x.message)}</span></div></div>`,
      )
      .join("");
    $("#startRunBtn").disabled = !data.ready;
    $("#startHeadline").textContent = data.ready
      ? "准备完成，可以开始正式分析"
      : "仍有必要项未完成";
    $("#startChecklist").textContent = data.ready
      ? "系统将执行追踪、号码识别、事件分析、沙盘/高光/球员卡与报告生成。"
      : "根据上方检查项完成必要配置。";
  }

  function updateCalibLists() {
    const imported = state.calib.imported,
      showImported =
        imported &&
        !state.calib.imagePoints.length &&
        !state.calib.validations.length;
    const importedPoints = showImported ? imported.image_points || [] : [],
      importedWorld = showImported ? imported.world_points_m || [] : [],
      importedValidations = showImported
        ? imported.validation_segments || []
        : [];
    $("#fitPointsList").innerHTML =
      state.calib.imagePoints
        .map(
          (p, i) =>
            `<div class="point-row"><span><b>P${i + 1}</b> 图像(${p[0].toFixed(0)},${p[1].toFixed(0)}) → 球场(${state.calib.worldPoints[i][0]},${state.calib.worldPoints[i][1]})</span><button data-rm-fit="${i}">×</button></div>`,
        )
        .join("") ||
      importedPoints
        .map((p, i) => {
          const w = importedWorld[i] || [];
          return `<div class="point-row imported"><span><b>P${i + 1}</b> 图像(${Number(p[0]).toFixed(0)},${Number(p[1]).toFixed(0)}) → 球场(${w[0] ?? "—"},${w[1] ?? "—"})m</span><em>已导入</em></div>`;
        })
        .join("") ||
      '<div class="point-row">尚未选择标定点</div>';
    $("#validationList").innerHTML =
      state.calib.validations
        .map(
          (v, i) =>
            `<div class="point-row"><span><b>V${i + 1}</b> 独立线段 → ${v.length_m}m</span><button data-rm-val="${i}">×</button></div>`,
        )
        .join("") ||
      importedValidations
        .map(
          (v, i) =>
            `<div class="point-row imported"><span><b>V${i + 1}</b> ${Number(v.known_length_m || 0).toFixed(2)}m${v.absolute_error_m != null ? ` · 误差 ${Number(v.absolute_error_m).toFixed(3)}m` : ""}</span><em>${v.passed === false ? "未通过" : "已验证"}</em></div>`,
        )
        .join("") ||
      '<div class="point-row">尚未选择独立验证线段</div>';
    $$("[data-rm-fit]").forEach((b) =>
      b.addEventListener("click", () => {
        const i = Number(b.dataset.rmFit);
        state.calib.imagePoints.splice(i, 1);
        state.calib.worldPoints.splice(i, 1);
        updateCalibLists();
        drawCalibOverlay();
      }),
    );
    $$("[data-rm-val]").forEach((b) =>
      b.addEventListener("click", () => {
        state.calib.validations.splice(Number(b.dataset.rmVal), 1);
        updateCalibLists();
        drawCalibOverlay();
      }),
    );
  }
  function calibImageGeometry() {
    const img = $("#calibImage"),
      wrap = img.parentElement.getBoundingClientRect(),
      sourceW = state.calib.sourceW || img.naturalWidth,
      sourceH = state.calib.sourceH || img.naturalHeight;
    if (!sourceW || !sourceH) return null;
    const scale = Math.min(wrap.width / sourceW, wrap.height / sourceH),
      width = sourceW * scale,
      height = sourceH * scale;
    return {
      wrap,
      sourceW,
      sourceH,
      width,
      height,
      left: (wrap.width - width) / 2,
      top: (wrap.height - height) / 2,
      scale,
    };
  }
  function drawCalibOverlay() {
    const img = $("#calibImage"),
      svg = $("#calibOverlay"),
      g = calibImageGeometry();
    if (!img.naturalWidth || !g) return;
    svg.setAttribute("viewBox", `0 0 ${g.wrap.width} ${g.wrap.height}`);
    let h = "";
    const visual = state.calib.dynamicVisual;
    if (visual?.accepted) {
      (visual.pitch_lines || []).forEach((line) => {
        const points = (line.points || [])
          .map((p) => `${g.left + p[0] * g.scale},${g.top + p[1] * g.scale}`)
          .join(" ");
        if (points)
          h += `<polyline class="dynamic-pitch-line ${esc(line.kind || "guide")}" points="${points}"/>`;
      });
    }
    const imported = state.calib.imported,
      current = Number($("#calibFrame").value),
      atImported = imported && Number(imported.frame_index) === current;
    const showImported =
        atImported &&
        !state.calib.imagePoints.length &&
        !state.calib.validations.length,
      points = showImported
        ? imported.image_points || []
        : state.calib.imagePoints,
      world = showImported
        ? imported.world_points_m || []
        : state.calib.worldPoints,
      validations = showImported
        ? imported.validation_segments || []
        : state.calib.validations;
    points.forEach((p, i) => {
      const x = g.left + p[0] * g.scale,
        y = g.top + p[1] * g.scale,
        w = world[i] || [],
        label =
          showImported && w.length >= 2
            ? `P${i + 1} · ${w[0]},${w[1]}m`
            : `P${i + 1}`,
        right = x > g.left + g.width * 0.72,
        tx = right ? x - 11 : x + 11,
        anchor = right ? "end" : "start";
      h += `<circle cx="${x}" cy="${y}" r="8" fill="#3478ff" stroke="#fff" stroke-width="2"/><text x="${tx}" y="${y + 4}" text-anchor="${anchor}" fill="#fff" font-size="11" font-weight="800" paint-order="stroke" stroke="#07101b" stroke-width="3">${label}</text>`;
    });
    validations.forEach((v, i) => {
      const x1 = g.left + v.p1[0] * g.scale,
        y1 = g.top + v.p1[1] * g.scale,
        x2 = g.left + v.p2[0] * g.scale,
        y2 = g.top + v.p2[1] * g.scale,
        label = `V${i + 1}${v.known_length_m || v.length_m ? ` · ${v.known_length_m || v.length_m}m` : ""}`;
      h += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#f39a42" stroke-width="3" stroke-dasharray="6 4"/><circle cx="${x1}" cy="${y1}" r="6" fill="#f39a42"/><circle cx="${x2}" cy="${y2}" r="6" fill="#f39a42"/><text x="${(x1 + x2) / 2 + 8}" y="${(y1 + y2) / 2 - 8}" fill="#ffd09a" font-size="11" font-weight="800" paint-order="stroke" stroke="#07101b" stroke-width="3">${label}</text>`;
    });
    if (state.calib.validationDraft.length) {
      const p = state.calib.validationDraft[0],
        x = g.left + p[0] * g.scale,
        y = g.top + p[1] * g.scale;
      h += `<circle cx="${x}" cy="${y}" r="7" fill="#f39a42" stroke="#fff" stroke-width="2"/>`;
    }
    svg.innerHTML = h;
  }
  function onCalibImageClick(e) {
    const img = $("#calibImage"),
      g = calibImageGeometry();
    if (!state.calib.clickMode || !img.naturalWidth || !g) return;
    const left = g.wrap.left + g.left,
      top = g.wrap.top + g.top;
    if (
      e.clientX < left ||
      e.clientX > left + g.width ||
      e.clientY < top ||
      e.clientY > top + g.height
    )
      return;
    const x = ((e.clientX - left) / g.width) * g.sourceW,
      y = ((e.clientY - top) / g.height) * g.sourceH;
    if (state.calib.clickMode === "fit") {
      if (state.calib.imagePoints.length >= 8) {
        toast("一个锚点最多 8 对对应点", "error");
        return;
      }
      state.calib.imagePoints.push([x, y]);
      state.calib.worldPoints.push([
        Number($("#worldX").value),
        Number($("#worldY").value),
      ]);
      state.calib.clickMode = null;
      $("#calibHint").textContent = "标定点已添加，可继续输入下一个球场坐标。";
    } else {
      state.calib.validationDraft.push([x, y]);
      if (state.calib.validationDraft.length === 2) {
        state.calib.validations.push({
          name: `validation_${state.calib.validations.length + 1}`,
          p1: state.calib.validationDraft[0],
          p2: state.calib.validationDraft[1],
          length_m: Number($("#validationLength").value),
        });
        state.calib.validationDraft = [];
        state.calib.clickMode = null;
        $("#calibHint").textContent = "独立验证线段已添加。";
      } else $("#calibHint").textContent = "请点击验证线段的第二个端点。";
    }
    updateCalibLists();
    drawCalibOverlay();
  }
  function clearAnchorDraft() {
    state.calib.imagePoints = [];
    state.calib.worldPoints = [];
    state.calib.validations = [];
    state.calib.validationDraft = [];
    state.calib.clickMode = null;
    updateCalibLists();
    drawCalibOverlay();
  }
  function loadCalibrationFrame(frameIndex, fromImported = false) {
    if (!state.project?.video) return;
    clearTimeout(state.calib.visualTimer);
    state.calib.dragPendingFrame = null;
    const fi = setCalibFrameUi(
        frameIndex,
        fromImported ? "已上传标定帧" : "视频帧",
      ),
      img = $("#calibImage"),
      video = $("#calibScrubVideo"),
      wrap = img.parentElement,
      token = ++state.calib.frameToken;
    state.calib.scrubbing = false;
    wrap.style.aspectRatio = `${state.project.video.width || 16} / ${state.project.video.height || 9}`;
    const loader = new Image();
    loader.decoding = "async";
    loader.onload = () => {
      if (token !== state.calib.frameToken) return;
      let done = false;
      const reveal = () => {
        if (done || token !== state.calib.frameToken) return;
        done = true;
        state.calib.sourceW = state.project.video.width;
        state.calib.sourceH = state.project.video.height;
        video.classList.remove("active");
        img.classList.remove("scrub-hidden");
        $("#calibHint").textContent = fromImported
          ? "蓝色为基准点，橙色为验证线；青色球场线是这一帧的动态映射效果。"
          : state.project.calibration?.status === "ready"
            ? "拖动下方时间轴可逐帧检查青色球场线是否稳定贴合。"
            : "选择“点选标定点”，再点击画面中的对应场地点。";
        drawCalibOverlay();
      };
      img.onload = reveal;
      img.src = loader.src;
      if (img.complete) requestAnimationFrame(reveal);
    };
    loader.src = calibrationFrameUrl(fi);
    loadDynamicFrameVisual(fi);
  }
  async function saveAnchor() {
    if (state.calib.imagePoints.length < 4) {
      toast("每个视角锚点至少需要 4 对对应点", "error");
      return;
    }
    if (!state.calib.validations.length) {
      toast("每个锚点至少需要 1 条独立验证线段", "error");
      return;
    }
    const payload = {
      frame_index: Number($("#calibFrame").value),
      image_points: state.calib.imagePoints,
      world_points: state.calib.worldPoints,
      validation_segments: state.calib.validations,
      field_length_m: Number($("#fieldLength").value),
      field_width_m: Number($("#fieldWidth").value),
      tolerance_m: Number($("#calibTolerance").value),
    };
    try {
      const c = await api(
        `/api/projects/${state.project.id}/calibration/anchors`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
      const last = (c.anchors || []).find(
        (a) => a.frame_index === payload.frame_index,
      );
      if (last?.passed) {
        toast("视角锚点已通过独立尺度验证", "success");
        clearAnchorDraft();
      } else toast("该锚点未通过独立验证，请调整点位", "error");
      await loadProject(state.project.id);
    } catch (e) {
      toast(e.message, "error");
    }
  }

  function renderProgress() {
    const p = state.project;
    if (!p) {
      return;
    }
    const pipe = p.pipeline || {};
    const progress = Number(pipe.progress) || 0;
    $("#progressRing").style.setProperty("--progress", `${progress}%`);
    $("#progressPercent").textContent = `${progress}%`;
    $("#progressTitle").textContent =
      pipe.state === "complete"
        ? "分析完成"
        : pipe.state === "running"
          ? "正在分析比赛"
          : pipe.state === "queued"
            ? "等待分析资源"
            : pipe.state === "failed"
              ? "分析中断"
              : "等待开始";
    $("#progressMessage").textContent = pipe.message || "等待开始";
    $("#runMeta").innerHTML =
      `<span>尝试 #${pipe.attempt || 0}</span><span>开始 ${fmtDate(pipe.started_at)}</span><span>结束 ${fmtDate(pipe.finished_at)}</span>`;
    $("#progressSteps").innerHTML = (pipe.steps || [])
      .map(
        (s, i) =>
          `<div class="progress-step ${s.state || ""}"><div class="step-top"><div class="step-icon">${s.state === "complete" ? "✓" : s.state === "failed" ? "!" : i + 1}</div><span class="state-chip ${s.state === "complete" ? "ok" : s.state === "failed" ? "bad" : "neutral"}">${s.state === "complete" ? "完成" : s.state === "running" ? `${s.progress || 0}%` : s.state === "failed" ? "失败" : "等待"}</span></div><h4>${esc(s.label)}</h4><p>${esc(s.message || s.hint || "")}</p><div class="bar"><i style="width:${s.state === "complete" ? 100 : s.progress || 0}%"></i></div></div>`,
      )
      .join("");
    $("#cancelBtn").classList.toggle("hidden", pipe.state !== "running");
    $("#retryBtn").classList.toggle(
      "hidden",
      !["failed", "interrupted", "cancelled"].includes(pipe.state),
    );
    $("#runHistory").innerHTML =
      (p.run_history || [])
        .slice()
        .reverse()
        .map(
          (r) =>
            `<div class="history-row"><div><b>运行 #${r.attempt || ""}</b><span>从 ${{ tracking: "追踪", jersey: "号码识别", events: "事件检测", report: "报告生成" }[r.from_step] || r.from_step}</span></div><span>${esc(r.state)}</span><span>${fmtDate(r.started_at)}</span></div>`,
        )
        .join("") ||
      '<div class="history-row"><span>尚无正式运行记录</span></div>';
  }
  function managePolling() {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
    const running =
      state.project?.pipeline?.state === "running" ||
      state.project?.calibration?.status === "building";
    if (!running) return;
    state.pollTimer = setInterval(async () => {
      try {
        const oldPipe = state.project.pipeline?.state,
          oldCal = state.project.calibration?.status;
        await loadProject(state.project.id, false);
        renderProjectCards();
        if (
          oldCal === "building" &&
          state.project.calibration?.status === "ready"
        )
          toast("全片动态标定已通过检查", "success");
        if (
          oldPipe === "running" &&
          state.project.pipeline?.state === "complete"
        ) {
          toast("分析完成，结果中心已就绪", "success");
          await loadResults();
          switchPage("results");
        }
      } catch {}
    }, 1800);
  }

  async function loadResults() {
    const p = state.project;
    if (!p) return;
    const complete =
      p.kind === "demo" ||
      p.pipeline?.state === "complete" ||
      p.status === "complete";
    $("#resultEmpty").classList.toggle("hidden", complete);
    $("#resultContent").classList.toggle("hidden", !complete);
    if (!complete) return;
    const id = p.id;
    const base = await Promise.all([
      api(`/api/projects/${id}/overview`),
      api(`/api/projects/${id}/pitch`),
      api(`/api/projects/${id}/events?limit=800`),
      api(`/api/projects/${id}/highlights`),
      api(`/api/projects/${id}/players`),
      api(`/api/projects/${id}/quality`),
      api(`/api/projects/${id}/files`),
    ]);
    const [ov, pitch, events, hs, ps, q, files] = base;
    let passReview = null,
      identityReview = null,
      assessmentReview = null;
    if (p.kind !== "demo") {
      [passReview, identityReview, assessmentReview] = await Promise.all([
        api(`/api/projects/${id}/reviews/passes`),
        api(`/api/projects/${id}/reviews/identities`),
        api(`/api/projects/${id}/reviews/assessments`),
      ]);
    }
    state.results = {
      overview: ov,
      pitch,
      events,
      highlights: hs,
      players: ps,
      quality: q,
      files,
      passReview,
      identityReview,
      assessmentReview,
    };
    state.playerIndex = Math.min(state.playerIndex, Math.max(0, ps.length - 1));
    $("#resultProjectName").textContent = p.name;
    const m = p.match || {};
    $("#resultSubtitle").textContent =
      `${m.home_team || "主队"} vs ${m.away_team || "客队"} · ${m.competition || "比赛分析"} · 比赛 → 空间 → 事件 → 球员 → 质检 → 报告 → 导出`;
    const zip = `/api/projects/${id}/export.zip`;
    $("#downloadZip").href = zip;
    $("#archiveDownload").href = zip;
    $("#openReportBtn").href = `/api/projects/${id}/report`;
    $("#reportFrame").src = `/api/projects/${id}/report`;
    $("#replayVideoLink").href = `/api/projects/${id}/replay.mp4`;
    $("#replayVideoLink").classList.toggle(
      "hidden",
      !ov.summary?.result_sections?.replay_video,
    );
    renderOverview();
    renderEvents();
    renderHighlights();
    renderPlayers();
    renderQuality();
    renderFiles();
  }
  function metricCard(icon, label, value, unit) {
    return `<div class="metric-card"><div class="metric-icon">${icon}</div><span>${label}</span><b>${value}</b><small>${unit || ""}</small></div>`;
  }
  function renderOverview() {
    const o = state.results.overview;
    if (!o) return;
    const s = o.summary,
      q = state.results.quality || {},
      c = q.calibration || {},
      pr = q.pass_review || {},
      ir = q.identity_review || {},
      pa = q.player_assessment || {};
    const ratio = c.validation?.accepted_ratio;
    $("#overviewStory").innerHTML =
      `<div class="insight-copy"><span class="badge green">比赛智能分析</span><h3>本场结果已按“空间 → 事件 → 球员 → 复核”形成完整分析视图。</h3><p>当前形成 <b>${s.candidate_ids || 0}</b> 个技术 ID 的候选轨迹、<b>${s.pass_candidates || 0}</b> 条主动传球候选和 <b>${s.confirmed_numbers || 0}</b> 个确认号码。未完成人工确认的身份、传球和八维能力会继续明确标记为候选或待复核。</p></div><div class="insight-status-grid"><div class="insight-status ${c.status === "ready" ? "ok" : ""}"><span>空间基础</span><b>${c.status === "ready" ? "动态标定通过" : "待检查"}</b><small>${ratio != null ? `有效覆盖 ${(ratio * 100).toFixed(1)}%` : "米制结果以标定状态为准"}</small></div><div class="insight-status ${pr.status === "complete" ? "ok" : ""}"><span>传球验收</span><b>${pr.status === "complete" ? "人工复核完成" : "仍需人工复核"}</b><small>${pr.labeled || 0}/${pr.sample_size || pr.total || 0} 条已标注</small></div><div class="insight-status ${ir.confirmed > 0 ? "ok" : ""}"><span>真实身份</span><b>${ir.confirmed || 0} 个已确认</b><small>技术 ID 与真实球员分层保存</small></div><div class="insight-status ${pa.confirmed > 0 ? "ok" : ""}"><span>能力卡</span><b>${pa.confirmed || 0} 个已评估</b><small>未确认时不生成假能力分</small></div></div>`;
    $("#overviewMetrics").innerHTML = [
      metricCard("◎", "候选球员轨迹", s.candidate_ids, "技术 ID"),
      metricCard("↝", "主动传球候选", s.pass_candidates, "条"),
      metricCard("◉", "稳定球权片段", s.possession_intervals, "段"),
      metricCard("↯", "最高速度", Number(s.peak_speed_mps).toFixed(2), "m/s"),
      metricCard(
        "⌁",
        "候选跑动总量",
        (Number(s.total_distance_m) / 1000).toFixed(1),
        "km",
      ),
      metricCard("#", "确认号码", s.confirmed_numbers, "个"),
    ].join("");
    drawOverviewPitch();
    renderTeamComparison();
    renderLeaders();
    $("#overviewTimeline").innerHTML =
      (o.timeline || [])
        .slice(0, 12)
        .map(
          (e) =>
            `<div class="timeline-row"><span class="timeline-time">${fmtTime(e.time_sec)}</span><i class="timeline-dot ${e.type}"></i><div class="timeline-main"><b>${esc(e.label)}</b><span>${e.from_id >= 0 ? `ID ${e.from_id}${e.to_id >= 0 ? ` → ID ${e.to_id}` : ""}` : ""} ${e.team ? `· ${esc(e.team)}` : ""}</span></div><span class="timeline-tag">${e.distance_m != null ? `${Number(e.distance_m).toFixed(1)}m` : ""}</span></div>`,
        )
        .join("") || '<div class="timeline-row"><span>暂无事件</span></div>';
  }
  function fieldGeom(canvas, field) {
    const w = canvas.width,
      h = canvas.height,
      pad = 48;
    const L = Number(field.length_m) || 45,
      W = Number(field.width_m) || 25;
    const scale = Math.min((w - 2 * pad) / L, (h - 2 * pad) / W);
    const fw = L * scale,
      fh = W * scale,
      ox = (w - fw) / 2,
      oy = (h - fh) / 2;
    return {
      w,
      h,
      pad,
      L,
      W,
      scale,
      fw,
      fh,
      ox,
      oy,
      pt: (x, y) => [ox + (Number(x) / L) * fw, oy + (Number(y) / W) * fh],
    };
  }
  function drawField(ctx, g) {
    ctx.clearRect(0, 0, g.w, g.h);
    ctx.fillStyle = "#06140f";
    ctx.fillRect(0, 0, g.w, g.h);
    const grad = ctx.createLinearGradient(g.ox, g.oy, g.ox + g.fw, g.oy);
    grad.addColorStop(0, "#0b3f2d");
    grad.addColorStop(0.5, "#10523a");
    grad.addColorStop(1, "#0a3d2b");
    ctx.fillStyle = grad;
    ctx.fillRect(g.ox, g.oy, g.fw, g.fh);
    ctx.strokeStyle = "rgba(233,250,241,.65)";
    ctx.lineWidth = 2;
    ctx.strokeRect(g.ox, g.oy, g.fw, g.fh);
    ctx.beginPath();
    ctx.moveTo(g.ox + g.fw / 2, g.oy);
    ctx.lineTo(g.ox + g.fw / 2, g.oy + g.fh);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(
      g.ox + g.fw / 2,
      g.oy + g.fh / 2,
      Math.min(g.fw, g.fh) * 0.105,
      0,
      Math.PI * 2,
    );
    ctx.stroke();
    const boxW = g.fw * 0.16,
      boxH = g.fh * 0.48;
    ctx.strokeRect(g.ox, g.oy + (g.fh - boxH) / 2, boxW, boxH);
    ctx.strokeRect(g.ox + g.fw - boxW, g.oy + (g.fh - boxH) / 2, boxW, boxH);
    ctx.fillStyle = "rgba(255,255,255,.85)";
    ctx.beginPath();
    ctx.arc(g.ox + g.fw / 2, g.oy + g.fh / 2, 3, 0, Math.PI * 2);
    ctx.fill();
  }
  function drawArrow(ctx, a, b, color, alpha = 0.7) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.stroke();
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]),
      len = 8;
    ctx.beginPath();
    ctx.moveTo(b[0], b[1]);
    ctx.lineTo(
      b[0] - len * Math.cos(ang - 0.45),
      b[1] - len * Math.sin(ang - 0.45),
    );
    ctx.lineTo(
      b[0] - len * Math.cos(ang + 0.45),
      b[1] - len * Math.sin(ang + 0.45),
    );
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  function drawOverviewPitch() {
    const data = state.results.pitch,
      canvas = $("#overviewPitch");
    if (!data || !canvas) return;
    const ctx = canvas.getContext("2d"),
      g = fieldGeom(canvas, data.field);
    drawField(ctx, g);
    (data.trails || []).forEach((t) => {
      const color = teamColor[t.team_id] || teamColor.unassigned;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 2;
      ctx.beginPath();
      t.points.forEach((p, i) => {
        const q = g.pt(p[0], p[1]);
        i ? ctx.lineTo(...q) : ctx.moveTo(...q);
      });
      ctx.stroke();
      const last = t.points.at(-1);
      if (last) {
        const q = g.pt(last[0], last[1]);
        ctx.globalAlpha = 1;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(...q, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "#fff";
        ctx.font = "8px sans-serif";
        ctx.fillText(String(t.global_id), q[0] + 7, q[1] + 3);
      }
      ctx.restore();
    });
    (data.passes || [])
      .slice(-45)
      .forEach((p) =>
        drawArrow(
          ctx,
          g.pt(...p.start),
          g.pt(...p.end),
          teamColor[p.team_id] || "#e8c25d",
          0.25,
        ),
      );
    const teams = [
      ...new Map((data.trails || []).map((t) => [t.team_id, t.team])).entries(),
    ];
    $("#pitchLegend").innerHTML =
      teams
        .map(
          ([id, label]) =>
            `<span class="legend-item"><i class="legend-dot" style="background:${teamColor[id] || teamColor.unassigned}"></i>${esc(label || id)}</span>`,
        )
        .join("") +
      `<span class="legend-item">轨迹来源：${esc(data.source_note || "")}</span>`;
  }
  function renderTeamComparison() {
    const rows = state.results.overview?.teams || [];
    const maxDist = Math.max(...rows.map((x) => Number(x.distance_m) || 0), 1),
      maxPass = Math.max(...rows.map((x) => Number(x.passes) || 0), 1);
    $("#teamComparison").innerHTML = rows.length
      ? rows
          .map(
            (t) =>
              `<div class="team-card"><div class="team-card-head"><b>${esc(t.label)}</b><span>${t.player_count} 个技术 ID</span></div><div class="team-stat"><div class="team-stat-head"><span>候选跑动</span><b>${(t.distance_m / 1000).toFixed(1)} km</b></div><div class="statbar"><i style="width:${(t.distance_m / maxDist) * 100}%;background:${teamColor[t.team_id] || "#4b8cff"}"></i></div></div><div class="team-stat"><div class="team-stat-head"><span>主动传球候选</span><b>${t.passes}</b></div><div class="statbar"><i style="width:${(t.passes / maxPass) * 100}%;background:${teamColor[t.team_id] || "#4b8cff"}"></i></div></div><div class="team-stat"><div class="team-stat-head"><span>稳定球权占比</span><b>${t.possession_share != null ? (t.possession_share * 100).toFixed(0) + "%" : "—"}</b></div><div class="statbar"><i style="width:${(t.possession_share || 0) * 100}%;background:${teamColor[t.team_id] || "#4b8cff"}"></i></div></div></div>`,
          )
          .join("")
      : '<div class="team-card"><span>暂无队伍数据</span></div>';
  }
  function renderLeaders() {
    const rows = state.results.overview?.leaders?.[state.leaderMode] || [];
    const label =
      state.leaderMode === "distance"
        ? (p) => `${(p.total_distance_m / 1000).toFixed(2)} km`
        : state.leaderMode === "speed"
          ? (p) => `${p.max_speed_mps.toFixed(2)} m/s`
          : (p) => `${p.sprint_count} 次`;
    $("#leaderList").innerHTML = rows
      .map(
        (p, i) =>
          `<div class="leader-row" data-player-id="${esc(p.player_id)}"><div class="leader-rank">${i + 1}</div><div><b>${esc(p.player_id)} · #${esc(p.jersey_number)}</b><span>${esc(p.team)} · ${esc(identityLabel(p.identity_status))}</span></div><div class="leader-value">${label(p)}</div></div>`,
      )
      .join("");
    $$("[data-player-id]").forEach((el) =>
      el.addEventListener("click", () => {
        const idx = state.results.players.findIndex(
          (p) => p.player_id === el.dataset.playerId,
        );
        if (idx >= 0) {
          state.playerIndex = idx;
          switchResult("players");
        }
      }),
    );
  }

  async function ensureReplay() {
    const video = $("#analysisVideo"),
      projectId = state.project.id,
      expectedDuration = Number(state.project.video?.duration_seconds || 0);
    // Video playback must not wait for the much larger tracking payload.
    video.preload = "auto";
    syncVideoSource(video, state.project, true);
    video.playbackRate = state.replay.speed;
    video.onloadedmetadata = () => {
      const canvas = $("#analysisOverlay");
      canvas.width = video.videoWidth || 960;
      canvas.height = video.videoHeight || 540;
      $("#replayDuration").textContent =
        `/ ${fmtTime(video.duration || expectedDuration)}`;
      drawReplay();
    };
    video.onended = () => setReplayPlaying(false);
    if (expectedDuration)
      $("#replayDuration").textContent = `/ ${fmtTime(expectedDuration)}`;
    if (state.replay.data) {
      drawReplay();
      return state.replay.data;
    }
    if (state.replay.loadingPromise) return state.replay.loadingPromise;
    $("#replayPlay").disabled = false;
    $("#replayRealtime").innerHTML =
      '<div class="realtime-head"><span>视频可以先播放</span><b>逐帧数据加载中</b></div><div class="realtime-empty">球员标记与实时数据将在索引读取完成后自动出现。</div>';
    state.replay.loadingPromise = api(
        `/api/projects/${state.project.id}/replay?start_frame=0&frame_count=${state.replay.windowSize}`,
      )
      .then((payload) => {
      if (state.project?.id !== projectId) return payload;
      const frameCount =
          Number(payload.total_frames) ||
          Number(state.project.video?.frame_count) ||
          Number(payload.frames?.at(-1)?.frame || 0) + 1,
        firstFrame = Number(payload.frames?.[0]?.frame || 0),
        lastFrame = Number(payload.frames?.at(-1)?.frame || firstFrame);
      payload.total_frames = frameCount;
      payload.window_start = Number(payload.window_start ?? firstFrame);
      payload.window_end = Number(payload.window_end ?? lastFrame + 1);
      payload.sampling_mode = payload.sampling_mode || "legacy_sampled";
      state.replay.data = payload;
      state.replay.windowCache = new Map([[payload.window_start, payload]]);
      state.replay.index = 0;
      state.replay.pitchTime = 0;
      state.replay.homographies = new Map();
      $("#replaySlider").max = Math.max(
        0,
        Number(payload.total_frames || 1) - 1,
      );
      $("#replayDuration").textContent =
        `/ ${fmtTime(payload.duration_seconds || expectedDuration)}`;
      renderReplayLegend();
      setReplayMode(state.replay.mode);
      drawReplay();
      if (payload.error) {
        $("#replayRealtime").innerHTML =
          `<div class="realtime-head"><span>${fmtTime(0)}</span><b>当前帧数据</b></div><div class="realtime-empty">逐帧追踪数据不可用：${esc(payload.error)}</div>`;
      } else if (payload.frames?.length) {
        updateReplayRealtime(payload.frames[0], 0, null);
      }
      if (payload.sampling_mode === "source_frame")
        loadReplayWindowForFrame(state.replay.windowSize, false).catch(() => {});
      if (state.replay.playing) replayTick();
      return payload;
    })
      .catch((error) => {
        if (state.project?.id === projectId) {
          $("#replayRealtime").innerHTML =
            '<div class="realtime-head"><span>视频仍可播放</span><b>轨迹数据读取失败</b></div><div class="realtime-empty">请稍后刷新重试，或检查分析结果是否完整。</div>';
          toast(`轨迹数据读取失败：${error.message}`, "error");
        }
        throw error;
      })
      .finally(() => {
        if (state.project?.id === projectId) state.replay.loadingPromise = null;
      });
    return state.replay.loadingPromise;
  }
  async function loadReplayWindowForFrame(frameNumber, activate = true) {
    if (state.replay.data?.sampling_mode !== "source_frame")
      return state.replay.data;
    const size = state.replay.windowSize,
      total = Number(state.replay.data?.total_frames || 1),
      start = Math.max(0, Math.min(Math.floor(frameNumber / size) * size, total - 1));
    if (state.replay.windowCache.has(start)) {
      const cached = state.replay.windowCache.get(start);
      if (activate) state.replay.data = cached;
      return cached;
    }
    if (state.replay.windowLoading?.start === start) {
      const payload = await state.replay.windowLoading.promise;
      if (activate) state.replay.data = payload;
      return payload;
    }
    const promise = api(
      `/api/projects/${state.project.id}/replay?start_frame=${start}&frame_count=${size}`,
    ).then((payload) => {
      state.replay.windowCache.set(start, payload);
      while (state.replay.windowCache.size > 4)
        state.replay.windowCache.delete(state.replay.windowCache.keys().next().value);
      return payload;
    });
    state.replay.windowLoading = { start, promise };
    try {
      const payload = await promise;
      if (activate) state.replay.data = payload;
      return payload;
    } finally {
      if (state.replay.windowLoading?.start === start)
        state.replay.windowLoading = null;
    }
  }
  function keepReplayWindowReady(time) {
    const d = state.replay.data;
    if (!d) return;
    const frame = Math.max(0, Math.floor(time * Number(d.fps || 30))),
      start = Number(d.window_start || 0),
      end = Number(d.window_end || 0);
    if (frame < start || frame >= end) {
      loadReplayWindowForFrame(frame).then(drawReplay).catch(() => {});
      return;
    }
    if (end - frame <= Math.max(60, Number(d.fps || 30) * 8)) {
      loadReplayWindowForFrame(end, false).catch(() => {});
    }
  }
  function renderReplayLegend() {
    const labels = state.replay.data?.team_labels || {};
    $("#replayLegend").innerHTML =
      Object.entries(labels)
        .map(
          ([id, label]) =>
            `<span class="legend-item"><i class="legend-dot" style="background:${teamColor[id] || teamColor.unassigned}"></i>${esc(label)}</span>`,
        )
        .join("") +
      '<span class="legend-item"><i class="legend-dot" style="background:#fff"></i>足球</span>';
  }
  function projectMetric(matrix, x, y) {
    if (!matrix) return null;
    const z = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2];
    if (!z || !Number.isFinite(z)) return null;
    return [
      (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / z,
      (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / z,
    ];
  }
  function replayFrameAtTime(time) {
    const frames = state.replay.data?.frames || [];
    if (!frames.length) return 0;
    let lo = 0,
      hi = frames.length - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (frames[mid].time_sec <= time) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }
  function interpolatedReplayFrame(time) {
    const d = state.replay.data,
      frames = d?.frames || [],
      i = replayFrameAtTime(time),
      a = frames[i],
      b = frames[Math.min(i + 1, frames.length - 1)];
    if (!a) return null;
    const span = Math.max(0.001, (b?.time_sec ?? a.time_sec) - a.time_sec),
      t = Math.max(0, Math.min(1, (time - a.time_sec) / span)),
      next = new Map((b?.players || []).map((p) => [p.id, p])),
      lerp = (x, y) => Number(x) + (Number(y) - Number(x)) * t;
    const players = (a.players || []).map((p) => {
      const q = next.get(p.id);
      if (!q) return p;
      const out = {
        ...p,
        x: lerp(p.x, q.x),
        y: lerp(p.y, q.y),
        speed: lerp(p.speed || 0, q.speed || 0),
      };
      if (p.image && q.image)
        out.image = [
          lerp(p.image[0], q.image[0]),
          lerp(p.image[1], q.image[1]),
        ];
      if (p.bbox && q.bbox)
        out.bbox = [0, 1, 2, 3].map((k) => lerp(p.bbox[k], q.bbox[k]));
      return out;
    });
    return { ...a, time_sec: time, players };
  }
  function observedBallAtTime(time) {
    const rows = state.replay.data?.ball_observations || [];
    if (!rows.length) return null;
    let lo = 0,
      hi = rows.length - 1;
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (rows[mid][0] < time) lo = mid + 1;
      else hi = mid;
    }
    const choices = [rows[lo], rows[Math.max(0, lo - 1)]].filter(Boolean),
      best = choices.sort(
        (a, b) => Math.abs(a[0] - time) - Math.abs(b[0] - time),
      )[0],
      tolerance = Math.max(0.055, 1.6 / (state.replay.data?.fps || 30));
    return best && Math.abs(best[0] - time) <= tolerance
      ? [best[1], best[2]]
      : null;
  }
  function drawDownTriangle(ctx, x, y, color, scale = 1) {
    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = "#06111d";
    ctx.lineWidth = 1.5 * scale;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - 8 * scale, y - 15 * scale);
    ctx.lineTo(x + 8 * scale, y - 15 * scale);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
  function drawOpenFootEllipse(ctx, x, y, rx, ry, color, width) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, -Math.PI / 4, (5 * Math.PI) / 4);
    ctx.stroke();
    ctx.restore();
  }
  function updateReplayRealtime(frame, time, activePass = null) {
    const root = $("#replayRealtime");
    if (!root || !frame) return;
    const speeds = (frame.players || []).map((p) => Number(p.speed) || 0),
      average = speeds.length
        ? speeds.reduce((sum, value) => sum + value, 0) / speeds.length
        : 0,
      peak = Math.max(0, ...speeds),
      owner = (frame.players || []).find(
        (p) => Number(p.id) === Number(frame.possession_id),
      ),
      observed = Boolean(frame.ball_image || frame.ball),
      eventText = activePass
        ? `传球：ID ${activePass.from} → ID ${activePass.to}`
        : owner
          ? `稳定持球：ID ${owner.id}`
          : "当前无稳定持球人";
    root.innerHTML = `<div class="realtime-head"><span>${fmtTime(time)}</span><b>当前帧数据</b></div><div class="realtime-grid"><div><span>源视频帧</span><b>${Number(frame.frame).toLocaleString()} / ${Number(state.replay.data?.total_frames || 0).toLocaleString()}</b></div><div><span>在场轨迹</span><b>${frame.players.length} 人</b></div><div><span>平均速度</span><b>${average.toFixed(2)} 米/秒</b></div><div><span>最高速度</span><b>${peak.toFixed(2)} 米/秒</b></div><div><span>持球人</span><b>${owner ? `ID ${owner.id}` : "无"}</b></div><div><span>足球观测</span><b>${observed ? "本帧检测到" : "本帧未检测到"}</b></div><div><span>播放数据</span><b>${Number(state.replay.data?.fps || 30).toFixed(1)} 帧/秒 · 逐帧</b></div></div><div class="realtime-event"><span>当前状态</span><b>${eventText}</b></div>`;
  }
  function drawLiveReplay() {
    const d = state.replay.data,
      video = $("#analysisVideo"),
      canvas = $("#analysisOverlay");
    if (!canvas) return;
    const time =
      Number.isFinite(video.currentTime) && video.readyState
        ? video.currentTime
        : d?.frames?.[state.replay.index]?.time_sec || 0;
    $("#replayTime").textContent = fmtTime(time);
    $("#replaySlider").value = Math.round(time * Number(d?.fps || 30));
    if (!d?.frames?.length) {
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (d?.error) {
        ctx.fillStyle = "rgba(20,30,40,.85)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#ff6b7a";
        ctx.font = "bold 20px Inter,Arial,sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("逐帧追踪数据加载失败", canvas.width / 2, canvas.height / 2 - 16);
        ctx.fillStyle = "#aab9ca";
        ctx.font = "14px Inter,Arial,sans-serif";
        ctx.fillText(d.error, canvas.width / 2, canvas.height / 2 + 16);
        ctx.textAlign = "start";
      }
      return;
    }
    keepReplayWindowReady(time);
    state.replay.index = replayFrameAtTime(time);
    const frame = interpolatedReplayFrame(time),
      ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const sourceW = Number(state.project?.video?.width) || canvas.width,
      sourceH = Number(state.project?.video?.height) || canvas.height,
      sx = canvas.width / sourceW,
      sy = canvas.height / sourceH,
      screen = (p) => (p ? [p[0] * sx, p[1] * sy] : null),
      byId = new Map((frame.players || []).map((p) => [p.id, p]));
    const activePass = (d.passes || []).find(
      (p) =>
        time >= p.time_sec &&
        time <= Math.max(p.end_time_sec, p.time_sec + 1.2),
    );
    if (activePass) {
      const a = screen(byId.get(activePass.from)?.image),
        b = screen(byId.get(activePass.to)?.image);
      if (a && b) {
        ctx.save();
        ctx.shadowColor = teamColor[activePass.team_id] || "#66f2c2";
        ctx.shadowBlur = 14;
        drawArrow(ctx, a, b, teamColor[activePass.team_id] || "#66f2c2", 1);
        ctx.restore();
      }
      $("#liveEventChip").classList.remove("hidden");
      $("#liveEventChip").innerHTML =
        `<span>传球</span><b>ID ${activePass.from} → ID ${activePass.to}</b>`;
    } else $("#liveEventChip").classList.add("hidden");
    (frame.players || []).forEach((p) => {
      const q = screen(p.image);
      if (
        !q ||
        q[0] < -100 ||
        q[0] > canvas.width + 100 ||
        q[1] < -100 ||
        q[1] > canvas.height + 100
      )
        return;
      const color = teamColor[p.team_id] || teamColor.unassigned,
        label = String(p.number || p.id),
        scale = Math.max(0.7, canvas.width / 960),
        box = p.bbox,
        boxWidth = box ? box[2] * sx : 36 * scale,
        rx = Math.max(13 * scale, Math.min(62 * scale, boxWidth * 0.68)),
        ry = Math.max(5 * scale, rx * 0.24),
        top = box
          ? [(box[0] + box[2] / 2) * sx, box[1] * sy]
          : [q[0], q[1] - 45 * scale];
      ctx.save();
      drawOpenFootEllipse(
        ctx,
        q[0],
        q[1],
        rx + 1.5 * scale,
        ry + 1 * scale,
        "rgba(255,255,255,.88)",
        1.5 * scale,
      );
      drawOpenFootEllipse(ctx, q[0], q[1], rx, ry, color, 3 * scale);
      const labelWidth = Math.max(26, 12 + label.length * 7) * scale;
      ctx.fillStyle = "rgba(239,247,244,.94)";
      ctx.fillRect(
        q[0] - labelWidth / 2,
        q[1] + ry + 3 * scale,
        labelWidth,
        17 * scale,
      );
      ctx.fillStyle = "#07130f";
      ctx.font = `800 ${10 * scale}px Inter,Arial,sans-serif`;
      ctx.textAlign = "center";
      ctx.fillText(label, q[0], q[1] + ry + 15 * scale);
      if (Number(frame.possession_id) === Number(p.id))
        drawDownTriangle(ctx, top[0], top[1] - 3 * scale, "#ff5b63", scale);
      ctx.restore();
    });
    const ball = screen(observedBallAtTime(time));
    if (ball)
      drawDownTriangle(
        ctx,
        ball[0],
        ball[1] - 4 * Math.max(0.7, canvas.width / 960),
        "#59f0b5",
        Math.max(0.7, canvas.width / 960),
      );
    const teams = state.results.overview?.teams || [];
    $("#livePossession").innerHTML = teams
      .filter((t) => t.possession_share != null)
      .slice(0, 2)
      .map(
        (t) =>
          `<div><i style="background:${teamColor[t.team_id] || teamColor.unassigned}"></i><span>${esc(t.label)} 控球</span><b>${(Number(t.possession_share) * 100).toFixed(1)}%</b></div>`,
      )
      .join("");
    updateReplayRealtime(frame, time, activePass);
  }
  function drawPitchReplay() {
    const d = state.replay.data,
      canvas = $("#replayCanvas");
    if (!d || !d.frames?.length) {
      if (canvas) {
        const c = canvas.getContext("2d");
        c.clearRect(0, 0, canvas.width, canvas.height);
        c.fillStyle = "#1a2535";
        c.fillRect(0, 0, canvas.width, canvas.height);
        c.textAlign = "center";
        if (d?.error) {
          c.fillStyle = "#ff6b7a";
          c.font = "bold 18px sans-serif";
          c.fillText("米制时序数据加载失败", canvas.width / 2, canvas.height / 2 - 20);
          c.fillStyle = "#7f98b3";
          c.font = "13px sans-serif";
          c.fillText(d.error, canvas.width / 2, canvas.height / 2 + 10);
        } else {
          c.fillStyle = "#8195ad";
          c.font = "18px sans-serif";
          c.fillText("暂无可回放的米制时序", canvas.width / 2, canvas.height / 2);
        }
        c.textAlign = "start";
      }
      return;
    }
    const base = d.frames[Math.min(state.replay.index, d.frames.length - 1)],
      time = Number.isFinite(state.replay.pitchTime)
        ? state.replay.pitchTime
        : base.time_sec,
      frame = interpolatedReplayFrame(time) || base,
      ctx = canvas.getContext("2d"),
      g = fieldGeom(canvas, d.field);
    keepReplayWindowReady(time);
    drawField(ctx, g);
    const activePass = (d.passes || []).find(
      (p) =>
        time >= p.time_sec &&
        time <= Math.max(p.end_time_sec, p.time_sec + 0.8),
    );
    if (activePass)
      drawArrow(
        ctx,
        g.pt(...activePass.start),
        g.pt(...activePass.end),
        teamColor[activePass.team_id] || "#ffe06b",
        0.95,
      );
    (frame.players || []).forEach((p) => {
      const q = g.pt(p.x, p.y),
        color = teamColor[p.team_id] || teamColor.unassigned,
        owner = Number(frame.possession_id) === Number(p.id);
      if (owner) {
        ctx.save();
        ctx.globalAlpha = 0.25;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(q[0], q[1], 18, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 0.8;
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.restore();
      }
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(q[0], q[1], 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 8px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(p.number || p.id, q[0], q[1] + 3);
      ctx.textAlign = "start";
      if (p.speed > 0) {
        ctx.fillStyle = "rgba(4,12,20,.75)";
        ctx.fillRect(q[0] - 19, q[1] + 12, 38, 13);
        ctx.fillStyle = "#d8e8f7";
        ctx.font = "8px sans-serif";
        ctx.textAlign = "center";
        ctx.fillText(`${p.speed.toFixed(1)}m/s`, q[0], q[1] + 22);
        ctx.textAlign = "start";
      }
    });
    if (frame.ball) {
      const q = g.pt(...frame.ball);
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(...q, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#222";
      ctx.stroke();
    } else {
      const ball = observedBallAtTime(time);
      if (ball) {
        const q = g.pt(...ball);
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(...q, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#222";
        ctx.stroke();
      }
    }
    const speeds = (frame.players || []).map((p) => Number(p.speed) || 0),
      avg = speeds.length
        ? speeds.reduce((a, b) => a + b, 0) / speeds.length
        : 0,
      peak = Math.max(0, ...speeds),
      owner = (frame.players || []).find(
        (p) => Number(p.id) === Number(frame.possession_id),
      );
    updateReplayRealtime(frame, time, activePass);
    $("#replayTime").textContent = fmtTime(time);
    $("#replaySlider").value = Math.round(time * d.fps);
  }
  function drawReplay() {
    if (state.replay.mode === "live") drawLiveReplay();
    else {
      drawLiveReplay();
      drawPitchReplay();
    }
  }
  function setReplayMode(mode) {
    const next = mode === "pitch" ? "pitch" : "live",
      video = $("#analysisVideo");
    if (next === "pitch" && video?.readyState)
      state.replay.pitchTime = video.currentTime;
    if (next === "live" && video && Number.isFinite(state.replay.pitchTime))
      video.currentTime = state.replay.pitchTime;
    state.replay.mode = next;
    setReplayPlaying(false);
    state.replay.index = replayFrameAtTime(state.replay.pitchTime || 0);
    $$("[data-replay-mode]").forEach((b) =>
      b.classList.toggle("active", b.dataset.replayMode === state.replay.mode),
    );
    const liveStage = $("#liveReplayStage");
    const pitchWrapper = $("#pitchCanvasWrapper");
    if (next === "pitch") {
      liveStage.classList.remove("hidden");
      pitchWrapper.classList.remove("hidden");
    } else {
      liveStage.classList.remove("hidden");
      pitchWrapper.classList.add("hidden");
    }
    drawReplay();
  }
  function replayTick() {
    if (!state.replay.playing) return;
    if (state.replay.mode === "live") {
      const video = $("#analysisVideo");
      if (state.replay.data?.frames?.length)
        state.replay.index = replayFrameAtTime(video.currentTime);
      drawLiveReplay();
      if (video.ended) {
        setReplayPlaying(false);
        return;
      }
      state.replay.timer = requestAnimationFrame(replayTick);
      return;
    }
    const video = $("#analysisVideo"),
      time = video?.readyState && Number.isFinite(video.currentTime)
        ? video.currentTime
        : state.replay.pitchStartTime +
          ((performance.now() - state.replay.pitchStartedAt) / 1000) *
            state.replay.speed;
    state.replay.pitchTime = time;
    if (state.replay.data?.frames?.length)
      state.replay.index = replayFrameAtTime(time);
    if (
      video?.ended ||
      time >= Number(state.replay.data?.duration_seconds || video?.duration || Infinity)
    ) {
      setReplayPlaying(false);
      drawReplay();
      return;
    }
    drawReplay();
    state.replay.timer = requestAnimationFrame(replayTick);
  }
  function setReplayPlaying(flag) {
    cancelAnimationFrame(state.replay.timer);
    clearTimeout(state.replay.timer);
    state.replay.playing = flag;
    $("#replayPlay").textContent = flag ? "❚❚" : "▶";
    const video = $("#analysisVideo");
    if (video) {
      if (flag) {
        video.playbackRate = state.replay.speed;
        if (
          state.replay.mode === "pitch" &&
          Number.isFinite(state.replay.pitchTime)
        )
          video.currentTime = state.replay.pitchTime;
        video.play().catch(() => {
          state.replay.playing = false;
          $("#replayPlay").textContent = "▶";
          toast("浏览器未能开始播放，请检查视频编码或重新点击播放。", "error");
        });
      } else video.pause();
    }
    if (flag && state.replay.mode === "pitch") {
      state.replay.pitchStartTime = Number.isFinite(state.replay.pitchTime)
        ? state.replay.pitchTime
        : state.replay.data?.frames?.[state.replay.index]?.time_sec || 0;
      state.replay.pitchStartedAt = performance.now();
    }
    if (flag) replayTick();
  }

  function renderEvents() {
    const rows = state.results.events || [];
    const type = $("#eventTypeFilter").value || "all",
      q = ($("#eventSearch").value || "").trim().toLowerCase();
    const filtered = rows.filter(
      (e) =>
        (type === "all" || e.type === type) &&
        (!q || JSON.stringify(e).toLowerCase().includes(q)),
    );
    $("#eventTable").innerHTML =
      `<table><thead><tr><th>时间</th><th>事件</th><th>队伍</th><th>球员</th><th>位移</th><th>结果状态</th></tr></thead><tbody>${filtered.map((e) => `<tr><td>${fmtTime(e.time_sec)}</td><td><span class="event-type-pill ${e.type}">${esc(e.label)}</span></td><td>${esc(e.team || "—")}</td><td>${e.from_id != null && e.from_id >= 0 ? `ID ${e.from_id}${e.to_id != null && e.to_id >= 0 ? ` → ID ${e.to_id}` : ""}` : "—"}</td><td>${e.distance_m != null ? Number(e.distance_m).toFixed(1) + " m" : "—"}</td><td>${esc(e.review || "")}</td></tr>`).join("") || '<tr><td colspan="6">没有符合筛选条件的事件</td></tr>'}</tbody></table>`;
  }
  function renderHighlights() {
    const priority = {
        goal_candidate: 0,
        counterpress_recovery: 1,
        shielding_under_pressure: 2,
      },
      rows = [...(state.results.highlights || [])].sort(
        (a, b) =>
          (priority[a.base_event_type] ?? 9) -
          (priority[b.base_event_type] ?? 9),
      );
    $("#highlightGrid").innerHTML = rows.length
      ? rows
          .map(
            (h) =>
              `<article class="highlight-card"><video controls preload="metadata" src="/api/projects/${state.project.id}/highlight/${h.index}"></video><div class="highlight-info"><div><b>${esc(String(h.name || "球员高光").replaceAll("TARGET", "目标球员"))}</b><span>${h.base_event_type ? esc(eventLabel(h.base_event_type)) : "目标球员标注片段"}${h.duration_seconds ? ` · ${Number(h.duration_seconds).toFixed(1)}秒` : ""}</span></div><span class="state-chip ${h.target_labeled ? "ok" : "neutral"}">${h.target_labeled ? "目标球员已标注" : "普通片段"}</span></div></article>`,
          )
          .join("")
      : '<div class="empty-state"><h3>暂无高光视频</h3><p>正式分析的报告生成阶段会自动输出带目标球员标记的片段。</p></div>';
  }
  function renderPlayers() {
    const rows = state.results.players || [],
      q = ($("#playerSearch").value || "").trim().toLowerCase();
    const filtered = rows
      .map((p, i) => ({ p, i }))
      .filter(({ p }) => !q || JSON.stringify(p).toLowerCase().includes(q));
    $("#playerList").innerHTML =
      filtered
        .map(
          ({ p, i }) =>
            `<div class="player-item ${i === state.playerIndex ? "active" : ""}" data-player-index="${i}"><div class="player-avatar">${esc(String(p.jersey_number || p.global_ids?.[0] || "?"))}</div><div><b>${esc(p.player_id)}</b><span>${esc(p.team)} · ${(p.global_ids || []).length > 1 ? `${p.global_ids.length} 个 ID 已关联` : esc(identityLabel(p.identity_status))}</span></div><span class="player-distance">${(p.total_distance_m / 1000).toFixed(2)}km</span></div>`,
        )
        .join("") || '<div class="point-row">没有匹配球员</div>';
    $$("[data-player-index]").forEach((x) =>
      x.addEventListener("click", () => {
        state.playerIndex = Number(x.dataset.playerIndex);
        renderPlayers();
      }),
    );
    renderPlayerCard()
      .then(renderPlayerEvidence)
      .catch(() => {});
  }
  function sourceToCompilation(manifest, sourceTime) {
    const row = (manifest?.intervals || []).find(
      (item) => sourceTime >= item.source_start && sourceTime < item.source_end,
    );
    return row ? row.compilation_start + sourceTime - row.source_start : null;
  }
  function drawPlayerCompilationTarget(video, canvas, manifest) {
    if (!video || !canvas || !manifest) return;
    canvas.width = video.videoWidth || 960;
    canvas.height = video.videoHeight || 540;
    const rows = manifest.boxes || [],
      time = video.currentTime;
    let lo = 0,
      hi = Math.max(0, rows.length - 1);
    while (lo < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if (rows[mid].time_sec < time) lo = mid + 1;
      else hi = mid;
    }
    const candidates = [rows[lo], rows[Math.max(0, lo - 1)]].filter(Boolean),
      row = candidates.sort(
        (a, b) => Math.abs(a.time_sec - time) - Math.abs(b.time_sec - time),
      )[0],
      ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!row || Math.abs(row.time_sec - time) > 0.12) return;
    const [x, y, w, h] = row.bbox,
      sourceW = Number(state.project?.video?.width) || 1920,
      sourceH = Number(state.project?.video?.height) || 1080,
      sx = canvas.width / sourceW,
      sy = canvas.height / sourceH,
      cx = (x + w / 2) * sx,
      foot = (y + h) * sy,
      rx = Math.max(14, w * sx * 0.68),
      ry = Math.max(5, rx * 0.24),
      scale = Math.max(0.7, canvas.width / 960);
    drawOpenFootEllipse(ctx, cx, foot, rx, ry, "#ffe06b", 4 * scale);
    drawDownTriangle(ctx, cx, y * sy - 3 * scale, "#ffe06b", scale);
    ctx.fillStyle = "rgba(5,15,24,.86)";
    ctx.fillRect(
      cx - 42 * scale,
      Math.max(4, y * sy - 42 * scale),
      84 * scale,
      20 * scale,
    );
    ctx.fillStyle = "#fff4b5";
    ctx.font = `800 ${10 * scale}px sans-serif`;
    ctx.textAlign = "center";
    ctx.fillText(
      `目标 ID ${row.global_id}`,
      cx,
      Math.max(18, y * sy - 28 * scale),
    );
    ctx.textAlign = "start";
  }
  async function renderPlayerEvidence() {
    const playerIndex = state.playerIndex,
      p = (state.results.players || [])[playerIndex],
      root = $("#playerEvidence");
    if (!root || !p) return;
    root.innerHTML = '<div class="point-row">正在读取球员事件与视频索引…</div>';
    const [events, manifest] = await Promise.all([
      api(`/api/projects/${state.project.id}/players/${playerIndex}/events`),
      api(
        `/api/projects/${state.project.id}/players/${playerIndex}/compilation-manifest`,
      ),
    ]);
    if (playerIndex !== state.playerIndex) return;
    state.replay.compilationManifest = manifest;
    root.innerHTML = `<div class="panel-title compact"><div><h3>球员证据与总视频</h3><p>视频中用开放式黄色脚圈和倒三角持续标记当前球员；事件可直接跳到总视频对应时刻。</p></div><div class="inline-actions"><button id="loadPlayerCompilation" class="btn secondary small">生成 / 加载球员总视频</button><button class="btn ghost small" data-open-player-report>进入正式报告</button></div></div><div class="player-evidence-grid"><div><div id="playerCompilationStage" class="player-compilation-stage"><video id="playerCompilation" controls controlslist="nofullscreen" preload="none"></video><canvas id="playerCompilationOverlay"></canvas><button id="playerCompilationFullscreen" class="player-fullscreen-btn" type="button">全屏复核</button></div><div class="visual-note">关联技术 ID：${esc((p.global_ids || []).join(", "))} · 总视频约 ${fmtTime(manifest.duration_seconds)}</div></div><div class="player-event-list">${
      events
        .map((e, index) => {
          const target = sourceToCompilation(manifest, Number(e.time_sec) || 0);
          const compTime = target != null ? fmtTime(target) : "—";
          return `<div class="player-event-row"><span class="event-time-original">${fmtTime(e.time_sec)}</span><span class="event-time-comp">总视频 ${compTime}</span><b>${esc(e.label)}</b><span>${e.from_id != null && e.from_id >= 0 ? `ID ${e.from_id}` : ""}${e.to_id != null && e.to_id >= 0 ? ` → ${e.to_id}` : ""}</span><button class="event-jump" data-compilation-time="${target == null ? "" : target.toFixed(3)}" ${target == null ? "disabled" : ""}>跳转</button></div>`;
        })
        .join("") || '<div class="point-row">该球员暂无候选事件</div>'
    }</div></div>`;
    const video = $("#playerCompilation"),
      canvas = $("#playerCompilationOverlay"),
      stage = $("#playerCompilationStage"),
      fullscreenButton = $("#playerCompilationFullscreen"),
      loadAt = (target = null) => {
        const seek = () => {
          if (target != null)
            video.currentTime = Math.max(
              0,
              Math.min(target, video.duration || target),
            );
          drawPlayerCompilationTarget(video, canvas, manifest);
          video.play().catch(() => {});
        };
        if (!video.src) {
          video.src = `/api/projects/${state.project.id}/players/${playerIndex}/compilation.mp4`;
          video.load();
          video.addEventListener("loadedmetadata", seek, { once: true });
        } else seek();
      };
    $("#loadPlayerCompilation").onclick = () => loadAt();
    $$("[data-compilation-time]").forEach(
      (button) =>
        (button.onclick = () => loadAt(Number(button.dataset.compilationTime))),
    );
    const paint = () => {
      drawPlayerCompilationTarget(video, canvas, manifest);
      if (!video.paused && !video.ended) requestAnimationFrame(paint);
    };
    video.addEventListener("play", paint);
    video.addEventListener("seeked", () =>
      drawPlayerCompilationTarget(video, canvas, manifest),
    );
    video.addEventListener("loadedmetadata", () =>
      drawPlayerCompilationTarget(video, canvas, manifest),
    );
    fullscreenButton.onclick = async () => {
      if (document.fullscreenElement === stage) await document.exitFullscreen();
      else await stage.requestFullscreen();
    };
    const syncFullscreen = () => {
      fullscreenButton.textContent =
        document.fullscreenElement === stage ? "退出全屏" : "全屏复核";
      requestAnimationFrame(() =>
        drawPlayerCompilationTarget(video, canvas, manifest),
      );
    };
    state.replay.compilationFullscreenAbort?.abort();
    state.replay.compilationFullscreenAbort = new AbortController();
    document.addEventListener("fullscreenchange", syncFullscreen, {
      signal: state.replay.compilationFullscreenAbort.signal,
    });
    $("[data-open-player-report]").onclick = () => switchResult("report");
  }
  function renderReportWorkflow() {
    const rows = state.results.players || [],
      select = $("#reportPlayerSelect");
    if (!select) return;
    select.innerHTML = rows
      .map(
        (p, i) =>
          `<option value="${i}" ${i === state.playerIndex ? "selected" : ""}>${esc(p.player_id)} · ID ${esc((p.global_ids || []).join(", "))}</option>`,
      )
      .join("");
    const apply = () => {
      state.playerIndex = Number(select.value) || 0;
      const base = `/api/projects/${state.project.id}/players/${state.playerIndex}/report`;
      $("#reportFrame").src = base;
      $("#openReportBtn").href = base;
      $("#downloadPlayerPdfBtn").href = `${base}.pdf`;
    };
    select.onchange = apply;
    apply();
  }
  async function renderPlayerCard() {
    const p = (state.results.players || [])[state.playerIndex],
      root = $("#playerCard");
    if (!p) {
      root.innerHTML = '<div class="empty-state">暂无球员数据</div>';
      return;
    }
    const formal = state.project?.kind !== "demo",
      gid = (p.global_ids || [])[0];
    const ir = state.results.identityReview || {},
      mapping = (ir.mappings || {})[String(gid)] || {};
    const ar = state.results.assessmentReview || {},
      assessment = (ar.assessments || {})[String(gid)] ||
        p.assessment || { scores: {}, status: "pending", note: "" },
      scores = assessment.scores || {};
    const teams = state.results.overview?.teams || [],
      roster = ir.roster || [],
      reviewData =
        formal && gid != null
          ? await Promise.all([
              api(
                `/api/projects/${state.project.id}/reviews/identities/${gid}/merge-candidates`,
              ),
              api(
                `/api/projects/${state.project.id}/reviews/player-report/${gid}`,
              ),
            ])
          : [{ candidates: [], rules: [] }, { fields: {} }],
      mergeReview = reviewData[0],
      reportFields = reviewData[1].fields || {};
    const mergeEditor =
      formal && gid != null
        ? `<details class="merge-editor"><summary>关联其他技术 ID <span>${(p.global_ids || []).length} 个已关联</span></summary><div class="merge-rules">${(mergeReview.rules || []).map((rule) => `<span>✓ ${esc(rule)}</span>`).join("")}</div><div class="merge-candidates">${(
            mergeReview.candidates || []
          )
            .filter((row) => row.global_id !== gid)
            .map(
              (row) =>
                `<label class="${row.compatible ? "" : "blocked"}"><input type="checkbox" data-link-gid="${row.global_id}" ${row.currently_linked && row.compatible ? "checked" : ""} ${row.compatible ? "" : "disabled"}><b>ID ${row.global_id}</b><span>${row.compatible ? row.team_id || "队伍待确认" : esc(row.reasons.join("；"))}</span></label>`,
            )
            .join("")}</div></details>`
        : "";
    const f = (key) => esc(reportFields[key] || ""),
      reportEditor =
        formal && gid != null
          ? `<div class="report-annotation-editor"><div class="panel-title compact"><div><h3>正式报告人工标注</h3><p>这些字段会直接进入该球员的单人 PDF；系统不会用跑动数据自动编造语义结论。</p></div><span class="state-chip neutral">人工填写</span></div><div class="report-fields three"><label>场上位置<input data-report-field="position" value="${f("position")}" placeholder="中前卫 / 前腰"></label><label>惯用脚<input data-report-field="preferred_foot" value="${f("preferred_foot")}" placeholder="右脚 / 左脚"></label><label>俱乐部 / 球队<input data-report-field="club" value="${f("club")}" placeholder="所属俱乐部"></label><label>球员标签<input data-report-field="nickname" value="${f("nickname")}" placeholder="枢纽 / 突击手"></label><label>潜力等级<input data-report-field="potential_grade" value="${f("potential_grade")}" placeholder="例如 A-"></label><label>潜力方向<input data-report-field="potential_direction" value="${f("potential_direction")}" placeholder="下一阶段目标"></label></div><label>一句话评语<textarea data-report-field="quote" placeholder="报告页顶部核心判断">${f("quote")}</textarea></label><div class="report-fields"><label>优势总结<textarea data-report-field="strengths_summary" placeholder="结合视频事件填写优势证据">${f("strengths_summary")}</textarea></label><label>提升方向 / 训练建议<textarea data-report-field="improvements_summary" placeholder="具体、可执行的提升建议">${f("improvements_summary")}</textarea></label></div><details class="report-advanced"><summary>风格、潜力与位置推荐详细字段</summary><div class="report-fields three"><label>风格标签<input data-report-field="style_tag" value="${f("style_tag")}" placeholder="空间连接型"></label><label>参考球员<input data-report-field="reference_player" value="${f("reference_player")}" placeholder="仅作风格参照"></label><label>下一目标<input data-report-field="next_target" value="${f("next_target")}" placeholder="下一阶段训练目标"></label><label>战术理解 0–100<input type="number" min="0" max="100" data-report-field="tactical_literacy" value="${f("tactical_literacy")}"></label><label>身体对抗 0–100<input type="number" min="0" max="100" data-report-field="physical_competition" value="${f("physical_competition")}"></label><label>天赋表现 0–100<input type="number" min="0" max="100" data-report-field="talent" value="${f("talent")}"></label></div><label>风格说明<textarea data-report-field="style_narrative" placeholder="解释风格判断及其证据">${f("style_narrative")}</textarea></label><div class="report-fields"><label>风格相似点（每行一条）<textarea data-report-field="similarities">${f("similarities")}</textarea></label><label>风格差异点（每行一条）<textarea data-report-field="differences">${f("differences")}</textarea></label></div>${[1, 2, 3].map((index) => `<div class="position-editor-row"><label>建议位置 ${index}<input data-report-field="position_${index}" value="${f(`position_${index}`)}"></label><label>匹配度<input type="number" min="0" max="100" data-report-field="position_${index}_fit" value="${f(`position_${index}_fit`)}"></label><label>依据<input data-report-field="position_${index}_description" value="${f(`position_${index}_description`)}"></label><label>结论<input data-report-field="position_${index}_verdict" value="${f(`position_${index}_verdict`)}"></label></div>`).join("")}</details><div class="report-fields"><label>给球员的话<textarea data-report-field="to_player">${f("to_player")}</textarea></label><label>给家长与教练<textarea data-report-field="to_family_and_coach">${f("to_family_and_coach")}</textarea></label></div><button id="saveReportAnnotationBtn" class="btn primary small">保存正式报告标注</button></div>`
          : "";
    const identityEditor =
      formal && gid != null
        ? `<div class="identity-editor"><div class="panel-title compact"><div><h3>真实身份确认</h3><p>算法只生成技术 ID；可在规则允许时关联时间上不重叠的轨迹片段。</p></div><span class="state-chip ${mapping.name ? "ok" : "neutral"}">${mapping.name ? "人工已确认" : "待确认"}</span></div>${roster.length ? `<label>从已上传名单选择<select id="identityRoster"><option value="">不关联名单</option>${roster.map((r) => `<option value="${r.index}" ${String(mapping.roster_index) === String(r.index) ? "selected" : ""}>${esc(r.name || "未命名")} ${r.jersey_number ? `#${esc(r.jersey_number)}` : ""}</option>`).join("")}</select></label>` : ""}<div class="identity-grid"><label>球员姓名<input id="identityName" value="${esc(mapping.name || "")}" placeholder="例如：张子豪" /></label><label>球衣号码<input id="identityNumber" value="${esc(mapping.jersey_number || "")}" placeholder="例如：9" /></label><label>所属队伍<select id="identityTeam"><option value="">保持算法分组</option>${teams.map((t) => `<option value="${esc(t.team_id)}" ${mapping.team_id === t.team_id ? "selected" : ""}>${esc(t.label)} (${esc(t.team_id)})</option>`).join("")}</select></label></div><label>复核备注<input id="identityNote" value="${esc(mapping.note || "")}" placeholder="可记录确认依据" /></label>${mergeEditor}<div class="inline-actions"><button id="saveIdentityBtn" class="btn primary small">保存身份确认</button><button id="clearIdentityBtn" class="btn ghost small">清除确认</button></div></div>`
        : "";
    const assessmentEditor =
      formal && gid != null
        ? `<div class="assessment-editor"><div class="panel-title compact"><div><h3>人工八维能力评估</h3><p>八维评分为人工确认的 0–100 分，不由系统虚构。8 项全部填写后才标记为“正式已评估”并进入报告。</p></div><span class="state-chip ${assessment.status === "confirmed" ? "ok" : "neutral"}">${assessment.status === "confirmed" ? "正式已评估" : assessment.status === "partial" ? "部分已填写" : "待评估"}</span></div><div class="assessment-grid">${assessmentDefs.map(([key, label]) => `<label><span>${label}</span><div><input type="range" min="0" max="100" step="1" data-assessment-range="${key}" value="${scores[key] ?? 50}"><input type="number" min="0" max="100" step="1" data-assessment-value="${key}" value="${scores[key] ?? ""}" placeholder="—"></div></label>`).join("")}</div><label class="assessment-note">评估备注<input id="assessmentNote" value="${esc(assessment.note || "")}" placeholder="例如：结合本场表现与教练人工观察" /></label><div class="inline-actions"><button id="saveAssessmentBtn" class="btn primary small">保存八维评估</button><button id="clearAssessmentBtn" class="btn ghost small">清除评估</button></div></div>`
        : "";
    const allPlayers = state.results.players || [];
    const rankOf = (key) => {
      const sorted = [...allPlayers].sort(
        (a, b) => (Number(b[key]) || 0) - (Number(a[key]) || 0),
      );
      const i = sorted.findIndex((x) => (x.global_ids || [])[0] === gid);
      return i >= 0 ? `${i + 1}/${sorted.length}` : "—";
    };
    const completeness = [
      mapping.name ? ["真实身份", "已确认", "ok"] : ["真实身份", "待确认", ""],
      assessment.status === "confirmed"
        ? ["八维能力", "已评估", "ok"]
        : ["八维能力", "待评估", ""],
      p.card_available
        ? ["正式球员卡", "已生成", "ok"]
        : ["正式球员卡", "基础数据", ""],
    ];
    const rankSummary = `<div class="player-rank-strip"><div><span>跑动排名</span><b>${rankOf("total_distance_m")}</b></div><div><span>速度排名</span><b>${rankOf("max_speed_mps")}</b></div><div><span>冲刺排名</span><b>${rankOf("sprint_count")}</b></div><div class="player-completeness">${completeness.map((x) => `<span class="${x[2]}">${x[0]} · ${x[1]}</span>`).join("")}</div></div>`;
    root.innerHTML = `<div class="player-card"><div class="player-card-head"><div class="player-identity"><div class="player-big-avatar">${esc(String(p.jersey_number || p.global_ids?.[0] || "?"))}</div><div><span class="badge blue">球员档案</span><h2>${esc(p.player_id)}</h2><p>${esc(p.team)} · 技术 ID ${esc((p.global_ids || []).join(", "))}</p></div></div><span class="identity-state">${esc(identityLabel(p.identity_status))}</span></div>${rankSummary}<div class="player-metrics"><div class="player-metric"><span>跑动距离</span><b>${(p.total_distance_m / 1000).toFixed(2)} km</b></div><div class="player-metric"><span>冲刺次数</span><b>${p.sprint_count} 次</b></div><div class="player-metric"><span>最高速度</span><b>${p.max_speed_mps.toFixed(2)} m/s</b></div><div class="player-metric"><span>有效追踪时间</span><b>${fmtTime(p.visible_time_sec)}</b></div></div><div class="player-tabs"><button class="player-tab active" data-player-tab="visuals">可视化</button><button class="player-tab" data-player-tab="mosaic">身份拼图</button>${formal && gid != null ? '<button class="player-tab" data-player-tab="identity">身份确认</button><button class="player-tab" data-player-tab="assessment">能力评估</button><button class="player-tab" data-player-tab="report">报告标注</button>' : ''}</div><div class="player-tab-content" data-tab-content="visuals"><div class="player-visuals"><div class="visual-card"><h4>位置热力图</h4><canvas id="heatmapCanvas" width="560" height="330"></canvas><div class="visual-note">按轨迹停留密度生成连续高斯热力图；红色表示高频区域。</div></div><div class="visual-card"><h4>八维能力雷达</h4><canvas id="radarCanvas" width="560" height="330"></canvas><div class="visual-note">${assessment.status === "confirmed" ? "展示已保存的人工八维正式评分。" : "未完成八维人工确认时保持“待评估”，不展示伪造能力分。"}</div></div></div></div><div class="player-tab-content hidden" data-tab-content="mosaic"><div class="visual-card mosaic-card"><h4>身份拼图</h4><div id="mosaicContainer" class="mosaic-container"><div class="mosaic-loading">正在加载身份拼图…</div></div><div class="visual-note">多帧采样拼图，快速确认技术 ID 对应的真实球员。点击可放大查看。</div></div></div>${formal && gid != null ? `<div class="player-tab-content hidden" data-tab-content="identity">${identityEditor}</div><div class="player-tab-content hidden" data-tab-content="assessment">${assessmentEditor}</div><div class="player-tab-content hidden" data-tab-content="report">${reportEditor}</div>` : ''}</div>`;
    const hm = await api(
      `/api/projects/${state.project.id}/players/${state.playerIndex}/heatmap`,
    );
    if (!(hm.points || []).length && !hm.image_url) {
      const identityKey = (p.global_ids || []).map(Number).sort((a, b) => a - b).join("_");
      hm.image_url = `/static/generated_heatmaps/${encodeURIComponent(state.project.id)}/ids_${identityKey}.png`;
    }
    drawHeatmap($("#heatmapCanvas"), hm);
    drawRadar($("#radarCanvas"), scores, assessment.status);
    $$("[data-player-tab]").forEach((btn) => {
      btn.onclick = () => {
        $$("[data-player-tab]").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        $$("[data-tab-content]").forEach((c) => c.classList.add("hidden"));
        const target = $(`[data-tab-content="${btn.dataset.playerTab}"]`);
        if (target) target.classList.remove("hidden");
      };
    });
    const mosaicContainer = $("#mosaicContainer");
    if (mosaicContainer) {
      api(`/api/projects/${state.project.id}/players/${state.playerIndex}/mosaic`).then((data) => {
        const mosaics = data.mosaics || [];
        if (!mosaics.length) {
          mosaicContainer.innerHTML = '<div class="mosaic-empty">该球员暂无身份拼图</div>';
          return;
        }
        mosaicContainer.innerHTML = mosaics.map((m) =>
          `<div class="mosaic-item" data-mosaic-url="${esc(m.mosaic_url)}" data-mosaic-id="${m.global_id}"><img src="${esc(m.mosaic_url)}" alt="ID ${m.global_id} 身份拼图" loading="lazy" /><div class="mosaic-meta"><span>ID ${m.global_id}</span><span>${m.sample_count} 帧采样</span><span>${fmtTime(m.visible_seconds)} 在场</span></div></div>`
        ).join('');
        mosaicContainer.querySelectorAll('.mosaic-item').forEach((item) => {
          item.onclick = () => {
            const url = item.dataset.mosaicUrl;
            const gid = item.dataset.mosaicId;
            if (!url) return;
            let overlay = document.getElementById('mosaicOverlay');
            if (!overlay) {
              overlay = document.createElement('div');
              overlay.id = 'mosaicOverlay';
              overlay.className = 'mosaic-overlay';
              overlay.innerHTML = '<div class="mosaic-overlay-backdrop"></div><div class="mosaic-overlay-content"><img /><div class="mosaic-overlay-info"></div><button class="mosaic-overlay-close">&times;</button></div>';
              document.body.appendChild(overlay);
              overlay.querySelector('.mosaic-overlay-backdrop').onclick = () => overlay.classList.remove('visible');
              overlay.querySelector('.mosaic-overlay-close').onclick = () => overlay.classList.remove('visible');
            }
            const img = overlay.querySelector('img');
            const info = overlay.querySelector('.mosaic-overlay-info');
            img.src = url;
            info.textContent = `技术 ID ${gid} · 身份拼图`;
            overlay.classList.add('visible');
          };
        });
      }).catch(() => {
        mosaicContainer.innerHTML = '<div class="mosaic-empty">身份拼图加载失败</div>';
      });
    }
    if (formal && gid != null) {
      const rosterEl = $("#identityRoster");
      if (rosterEl)
        rosterEl.onchange = () => {
          const r = roster.find((x) => String(x.index) === rosterEl.value);
          if (r) {
            $("#identityName").value = r.name || "";
            $("#identityNumber").value = r.jersey_number || "";
          }
        };
      $("#saveIdentityBtn").onclick = () => saveIdentity(gid, false);
      $("#clearIdentityBtn").onclick = () => saveIdentity(gid, true);
      $("#saveReportAnnotationBtn").onclick = () => saveReportAnnotation(gid);
      $$("[data-assessment-range]").forEach(
        (el) =>
          (el.oninput = () => {
            const n = $(
              `[data-assessment-value="${el.dataset.assessmentRange}"]`,
            );
            if (n) n.value = el.value;
          }),
      );
      $$("[data-assessment-value]").forEach(
        (el) =>
          (el.oninput = () => {
            const r = $(
              `[data-assessment-range="${el.dataset.assessmentValue}"]`,
            );
            if (r && el.value !== "")
              r.value = Math.max(0, Math.min(100, Number(el.value) || 0));
          }),
      );
      $("#saveAssessmentBtn").onclick = () => saveAssessment(gid, false);
      $("#clearAssessmentBtn").onclick = () => saveAssessment(gid, true);
    }
  }
  function drawHeatmap(canvas, data) {
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const displayW = canvas.clientWidth || canvas.parentElement?.clientWidth || 560;
    const displayH = Math.round(displayW * 330 / 560);
    canvas.width = displayW * dpr;
    canvas.height = displayH * dpr;
    ctx.scale(dpr, dpr);
    const g = fieldGeom({ width: displayW, height: displayH }, data.field);
    const length = Number(data.field?.length_m) || 45;
    const width = Number(data.field?.width_m) || 25;
    ctx.clearRect(0, 0, displayW, displayH);
    if (!(data.points || []).length && data.image_url) {
      ctx.fillStyle = "#2d7d34";
      ctx.fillRect(0, 0, displayW, displayH);
      ctx.fillStyle = "rgba(255,255,255,.72)";
      ctx.font = "14px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("正在读取球员热力图…", displayW / 2, displayH / 2);
      const image = new Image();
      image.onload = () => {
        ctx.clearRect(0, 0, displayW, displayH);
        ctx.fillStyle = "#2d7d34";
        ctx.fillRect(0, 0, displayW, displayH);
        const scale = Math.min(displayW / image.width, displayH / image.height),
          dw = image.width * scale,
          dh = image.height * scale;
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(image, (displayW - dw) / 2, (displayH - dh) / 2, dw, dh);
      };
      image.onerror = () => drawHeatmap(canvas, { ...data, image_url: null });
      image.src = data.image_url;
      return;
    }
    ctx.fillStyle = "#2d7d34";
    ctx.fillRect(0, 0, displayW, displayH);
    const gw = 360,
      gh = Math.max(140, Math.round((gw * g.fh) / g.fw)),
      values = new Float32Array(gw * gh),
      sigma = 8,
      radius = Math.ceil(sigma * 3),
      points = data.points || [];
    points.forEach((p) => {
      const gx = (Number(p[0]) / length) * (gw - 1),
        gy = (Number(p[1]) / width) * (gh - 1);
      if (!Number.isFinite(gx) || !Number.isFinite(gy)) return;
      const x0 = Math.max(0, Math.floor(gx - radius)),
        x1 = Math.min(gw - 1, Math.ceil(gx + radius)),
        y0 = Math.max(0, Math.floor(gy - radius)),
        y1 = Math.min(gh - 1, Math.ceil(gy + radius));
      for (let y = y0; y <= y1; y++)
        for (let x = x0; x <= x1; x++) {
          const dx = x - gx,
            dy = y - gy;
          values[y * gw + x] += Math.exp(
            -(dx * dx + dy * dy) / (2 * sigma * sigma),
          );
        }
    });
    const max = Math.max(0, ...values),
      layer = document.createElement("canvas");
    layer.width = gw;
    layer.height = gh;
    const lc = layer.getContext("2d"),
      image = lc.createImageData(gw, gh);
    for (let i = 0; i < values.length; i++) {
      let v = max ? Math.pow(values[i] / max, 0.62) : 0;
      if (v < 0.055) continue;
      let r, gc, b;
      if (v < 0.3) {
        const t = v / 0.3;
        r = 20;
        gc = 120 + 100 * t;
        b = 190 + 40 * t;
      } else if (v < 0.55) {
        const t = (v - 0.3) / 0.25;
        r = 20 + 225 * t;
        gc = 220 + 20 * t;
        b = 230 - 180 * t;
      } else if (v < 0.78) {
        const t = (v - 0.55) / 0.23;
        r = 245 + 10 * t;
        gc = 240 - 105 * t;
        b = 50 - 25 * t;
      } else {
        const t = (v - 0.78) / 0.22;
        r = 255 - 35 * t;
        gc = 135 - 105 * t;
        b = 25 + 5 * t;
      }
      const j = i * 4;
      image.data[j] = r;
      image.data[j + 1] = gc;
      image.data[j + 2] = b;
      image.data[j + 3] = Math.round(35 + 210 * v);
    }
    lc.putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(layer, g.ox, g.oy, g.fw, g.fh);
    ctx.strokeStyle = "rgba(255,255,255,.9)";
    ctx.lineWidth = 2;
    ctx.strokeRect(g.ox, g.oy, g.fw, g.fh);
    ctx.beginPath();
    ctx.moveTo(g.ox + g.fw / 2, g.oy);
    ctx.lineTo(g.ox + g.fw / 2, g.oy + g.fh);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(
      g.ox + g.fw / 2,
      g.oy + g.fh / 2,
      Math.min(g.fw, g.fh) * 0.105,
      0,
      Math.PI * 2,
    );
    ctx.stroke();
    const boxW = g.fw * 0.13,
      boxH = g.fh * 0.46;
    ctx.strokeRect(g.ox, g.oy + (g.fh - boxH) / 2, boxW, boxH);
    ctx.strokeRect(g.ox + g.fw - boxW, g.oy + (g.fh - boxH) / 2, boxW, boxH);
  }
  function drawRadar(canvas, scores = {}, status = "pending") {
    const ctx = canvas.getContext("2d"),
      cx = canvas.width / 2,
      cy = canvas.height / 2 + 2,
      R = 118,
      defs = assessmentDefs,
      vals = defs.map(([k]) =>
        scores[k] == null
          ? null
          : Math.max(0, Math.min(1, Number(scores[k]) / 100)),
      );
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#081522";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let ring = 1; ring <= 4; ring++) {
      ctx.beginPath();
      defs.forEach((_, i) => {
        const a = -Math.PI / 2 + (i * 2 * Math.PI) / defs.length,
          r = (R * ring) / 4,
          x = cx + Math.cos(a) * r,
          y = cy + Math.sin(a) * r;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.closePath();
      ctx.strokeStyle = "#243c58";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
    defs.forEach(([_, lab], i) => {
      const a = -Math.PI / 2 + (i * 2 * Math.PI) / defs.length,
        x = cx + Math.cos(a) * R,
        y = cy + Math.sin(a) * R;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(x, y);
      ctx.strokeStyle = "#203852";
      ctx.stroke();
      ctx.fillStyle = "#8ca2bb";
      ctx.font = "12px sans-serif";
      ctx.textAlign =
        Math.cos(a) > 0.3 ? "left" : Math.cos(a) < -0.3 ? "right" : "center";
      ctx.fillText(
        lab,
        cx + Math.cos(a) * (R + 22),
        cy + Math.sin(a) * (R + 22) + 4,
      );
    });
    if (status === "confirmed" && vals.every((v) => v != null)) {
      ctx.beginPath();
      vals.forEach((v, i) => {
        const a = -Math.PI / 2 + (i * 2 * Math.PI) / defs.length,
          x = cx + Math.cos(a) * R * v,
          y = cy + Math.sin(a) * R * v;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = "rgba(74,132,255,.19)";
      ctx.fill();
      ctx.strokeStyle = "#6b9cff";
      ctx.lineWidth = 2.4;
      ctx.stroke();
      vals.forEach((v, i) => {
        const a = -Math.PI / 2 + (i * 2 * Math.PI) / defs.length;
        ctx.fillStyle = "#a9c5ff";
        ctx.beginPath();
        ctx.arc(
          cx + Math.cos(a) * R * v,
          cy + Math.sin(a) * R * v,
          3.5,
          0,
          Math.PI * 2,
        );
        ctx.fill();
      });
    } else {
      ctx.fillStyle = "#7f95ae";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(
        status === "partial"
          ? "部分评分已保存 · 完成 8 项后生成正式雷达"
          : "待评估 · 不使用假分数",
        cx,
        cy + 5,
      );
      ctx.textAlign = "start";
    }
  }
  function renderQuality() {
    const q = state.results.quality;
    if (!q) return;
    const c = q.calibration || {},
      j = q.jersey || {},
      pr = q.pass_review || {},
      ir = q.identity_review || {},
      pa = q.player_assessment || {};
    const ratio = c.validation?.accepted_ratio,
      formal = state.project?.kind !== "demo",
      review = state.results.passReview || { rows: [] };
    const agree =
      review.agreement_rate != null
        ? `${(review.agreement_rate * 100).toFixed(1)}%`
        : "—";
    const teamRows = state.results.overview?.teams || [];
    const teamEditor = formal
      ? `<section class="review-section panel"><div class="panel-title compact"><div><h3>队伍语义确认</h3><p>系统不会猜“主队/客队”。请把算法分组明确映射成比赛中的实际队伍名称。</p></div></div><div class="team-map-grid">${teamRows.map((t) => `<label><span>${esc(t.team_id)} · ${t.player_count} 个技术 ID</span><input data-team-map="${esc(t.team_id)}" value="${esc(t.label || "")}" /></label>`).join("") || "<span>暂无队伍分组</span>"}</div><button id="saveTeamMapBtn" class="btn secondary small">保存队伍名称</button></section>`
      : "";
    const passTable = formal
      ? `<section class="review-section panel"><div class="panel-title compact"><div><h3>传球 20 条人工复核</h3><p>点击“查看视频”会在候选下方展开前后证据，再判断是否为传球。</p></div><span class="state-chip ${review.status === "complete" ? "ok" : "neutral"}">${review.labeled || 0}/${review.total || 0} · ${agree}</span></div><div class="pass-review-list">${(review.rows || []).map((r) => `<div class="pass-review-row"><div class="pass-review-main"><b>${fmtTime(r.time_sec)} · ID ${esc(r.from_global_id)} → ID ${esc(r.to_global_id)}</b><span>${esc(r.team_id || "未分组")} · ${Number(r.distance_m || 0).toFixed(1)}m</span></div><div class="review-choice"><button class="pass-video-button" data-pass-video="${esc(r.key)}" data-time="${Number(r.time_sec) || 0}" data-from="${esc(r.from_global_id)}" data-to="${esc(r.to_global_id)}">▶ 查看视频</button><button class="${r.human_is_pass === true ? "active yes" : ""}" data-pass-review="${esc(r.key)}" data-value="yes">✓ 是传球</button><button class="${r.human_is_pass === false ? "active no" : ""}" data-pass-review="${esc(r.key)}" data-value="no">× 不是</button><button data-pass-review="${esc(r.key)}" data-value="clear">清除</button></div><div class="pass-review-video hidden" data-pass-video-panel="${esc(r.key)}"><div class="pass-video-wrapper"><video controls preload="metadata" playsinline></video><canvas class="pass-video-overlay"></canvas></div><div><b>候选时刻 ${fmtTime(r.time_sec)}</b><span>默认播放候选前 3 秒至后 5 秒，可拖动进一步判断。</span><span class="pass-player-hint">传球方 ID ${esc(r.from_global_id)} → 接球方 ID ${esc(r.to_global_id)}</span></div></div></div>`).join("") || '<div class="point-row">当前没有可复核传球样本。</div>'}</div></section>`
      : "";
    $("#qualityContent").innerHTML =
      `<div class="quality-summary-grid"><div class="quality-card"><h3>动态标定</h3><div class="quality-row"><span>状态</span><b>${c.status === "ready" ? "通过" : "待检查"}</b></div><div class="quality-row"><span>视角锚点</span><b>${(c.anchors || []).length || c.validation?.anchor_count || "—"}</b></div><div class="quality-row"><span>有效覆盖</span><b>${ratio != null ? (ratio * 100).toFixed(1) + "%" : "—"}</b></div><div class="quality-row"><span>说明</span><b>${esc(c.message || "")}</b></div></div><div class="quality-card"><h3>号码识别</h3><div class="quality-row"><span>候选 ID</span><b>${j.total || 0}</b></div>${Object.entries(
        j.status_counts || {},
      )
        .map(
          ([k, v]) =>
            `<div class="quality-row"><span>${esc(k)}</span><b>${v}</b></div>`,
        )
        .join(
          "",
        )}</div><div class="quality-card"><h3>人工复核</h3><div class="quality-row"><span>传球验收</span><b>${pr.status === "complete" ? "已完成" : "待复核"}</b></div><div class="quality-row"><span>传球已标注</span><b>${pr.labeled || 0}/${pr.sample_size || 0}</b></div><div class="quality-row"><span>身份确认</span><b>${ir.confirmed || 0}/${ir.total || 0}</b></div><div class="quality-row"><span>八维评估</span><b>${pa.confirmed || 0}/${pa.total || 0}</b></div></div><div class="quality-card"><h3>跑动与事件</h3><div class="quality-row"><span>米制跑动</span><b>${q.running?.available ? "已生成" : "无"}</b></div><div class="quality-row"><span>分析质量报告</span><b>${q.analysis?.available ? "已生成" : "无"}</b></div><div class="quality-row"><span>候选身份数</span><b>${q.running?.identities || "—"}</b></div></div><div class="quality-card"><h3>身份质量审计</h3><div class="quality-row"><span>审计状态</span><b>${q.identity_audit?.available ? "已生成" : q.identity_audit?.status === "warning" ? "警告" : "未运行"}</b></div><div class="quality-row"><span>疑似污染 ID</span><b>${q.identity_audit?.ids_with_transitions || 0}</b></div><div class="quality-row"><span>切换候选</span><b>${q.identity_audit?.detected_transitions || 0}</b></div><div class="quality-row"><span>策略</span><b>只提示，不自动改 ID</b></div></div></div>${teamEditor}${passTable}<div class="quality-note">${esc(q.product_note || "")}</div>`;
    if (formal) {
      const teamBtn = $("#saveTeamMapBtn");
      if (teamBtn) teamBtn.onclick = saveTeamLabels;
      $$("[data-pass-review]").forEach(
        (b) =>
          (b.onclick = () =>
            savePassReview(b.dataset.passReview, b.dataset.value)),
      );
      $$("[data-pass-video]").forEach((button) => {
        button.onclick = () => {
          const panel = document.querySelector(
              `[data-pass-video-panel="${CSS.escape(button.dataset.passVideo)}"]`,
            ),
            video = panel?.querySelector("video"),
            canvas = panel?.querySelector(".pass-video-overlay"),
            opening = panel?.classList.contains("hidden"),
            candidateTime = Number(button.dataset.time) || 0,
            fromId = button.dataset.from || "",
            toId = button.dataset.to || "",
            clipStart = Math.max(0, candidateTime - 3),
            clipEnd = candidateTime + 5;
          $$(".pass-review-video:not(.hidden)").forEach((other) => {
            if (other !== panel) {
              other.classList.add("hidden");
              other.querySelector("video")?.pause();
            }
          });
          if (!panel || !video) return;
          panel.classList.toggle("hidden", !opening);
          button.textContent = opening ? "收起视频" : "▶ 查看视频";
          if (!opening) {
            video.pause();
            return;
          }
          const ensureReplayData = async () => {
            try {
              const fps = Number(state.project?.video?.fps || 30);
              const startFrame = Math.max(0, Math.floor(clipStart * fps));
              const frameCount = Math.ceil((clipEnd - clipStart + 2) * fps);
              const payload = await api(
                `/api/projects/${state.project.id}/replay?start_frame=${startFrame}&frame_count=${frameCount}`,
              );
              return payload;
            } catch { return null; }
          };
          ensureReplayData().then((clipData) => {
          const drawPassOverlay = () => {
            if (!canvas || !video.videoWidth) return;
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const time = video.currentTime;
            const d = clipData || state.replay.data;
            const sourceW = Number(state.project?.video?.width) || canvas.width;
            const sourceH = Number(state.project?.video?.height) || canvas.height;
            const sx = canvas.width / sourceW;
            const sy = canvas.height / sourceH;
            if (d?.frames?.length) {
              let frame = null;
              const frames = d.frames;
              for (let i = 0; i < frames.length; i++) {
                if (Math.abs(frames[i].time_sec - time) < 0.04) {
                  frame = frames[i];
                  break;
                }
              }
              if (!frame && frames.length > 1) {
                let lo = 0, hi = frames.length - 1;
                while (lo < hi) {
                  const mid = Math.floor((lo + hi) / 2);
                  if (frames[mid].time_sec < time) lo = mid + 1;
                  else hi = mid;
                }
                frame = frames[lo];
              }
              if (frame && frame.players) {
                frame.players.forEach((p) => {
                  const box = p.bbox;
                  if (box) {
                    const bx = box[0] * sx;
                    const by = box[1] * sy;
                    const bw = box[2] * sx;
                    const bh = box[3] * sy;
                    const isFrom = String(p.id) === String(fromId);
                    const isTo = String(p.id) === String(toId);
                    if (isFrom || isTo) {
                      ctx.strokeStyle = isFrom ? "#ffb43c" : "#3cb4ff";
                      ctx.lineWidth = 3;
                      ctx.strokeRect(bx, by, bw, bh);
                      const label = `ID ${p.id}`;
                      ctx.font = `bold ${Math.max(14, bh * 0.18)}px Inter,Arial,sans-serif`;
                      const tw = ctx.measureText(label).width;
                      ctx.fillStyle = isFrom ? "rgba(255, 180, 60, 0.9)" : "rgba(60, 180, 255, 0.9)";
                      ctx.fillRect(bx, by - 22, tw + 10, 20);
                      ctx.fillStyle = isFrom ? "#1a1000" : "#001a33";
                      ctx.fillText(label, bx + 5, by - 6);
                    } else {
                      ctx.strokeStyle = "rgba(150, 180, 200, 0.4)";
                      ctx.lineWidth = 1;
                      ctx.strokeRect(bx, by, bw, bh);
                    }
                  }
                  if (p.image) {
                    const qx = p.image[0] * sx;
                    const qy = p.image[1] * sy;
                    const isFrom = String(p.id) === String(fromId);
                    const isTo = String(p.id) === String(toId);
                    if (isFrom || isTo) {
                      const color = isFrom ? "#ffb43c" : "#3cb4ff";
                      ctx.beginPath();
                      ctx.arc(qx, qy, 8, 0, Math.PI * 2);
                      ctx.fillStyle = color;
                      ctx.fill();
                      ctx.strokeStyle = "#fff";
                      ctx.lineWidth = 2;
                      ctx.stroke();
                    }
                  }
                });
              }
            }
            const labelH = Math.max(24, canvas.height * 0.035);
            const fontSize = Math.max(12, canvas.height * 0.025);
            ctx.font = `bold ${fontSize}px Inter,Arial,sans-serif`;
            ctx.textAlign = "left";
            const fromLabel = `传球方 ID ${fromId}`;
            const fromW = ctx.measureText(fromLabel).width + 20;
            ctx.fillStyle = "rgba(255, 180, 60, 0.85)";
            ctx.fillRect(8, 8, fromW, labelH);
            ctx.fillStyle = "#1a1000";
            ctx.fillText(fromLabel, 18, 8 + labelH * 0.7);
            ctx.textAlign = "right";
            const toLabel = `接球方 ID ${toId}`;
            const toW = ctx.measureText(toLabel).width + 20;
            ctx.fillStyle = "rgba(60, 180, 255, 0.85)";
            ctx.fillRect(canvas.width - toW - 8, 8, toW, labelH);
            ctx.fillStyle = "#001a33";
            ctx.fillText(toLabel, canvas.width - 18, 8 + labelH * 0.7);
            ctx.textAlign = "start";
          };
          if (!video.src) {
            video.src = `/api/projects/${state.project.id}/preview-video#t=${clipStart.toFixed(3)},${clipEnd.toFixed(3)}`;
            video.load();
          }
          let passAnimFrame = null;
          const paintPass = () => {
            drawPassOverlay();
            if (!video.paused && !video.ended) passAnimFrame = requestAnimationFrame(paintPass);
          };
          const start = () => {
            video.currentTime = clipStart;
            drawPassOverlay();
            video.play().catch(() => {});
          };
          if (video.readyState >= 1) start();
          else video.addEventListener("loadedmetadata", start, { once: true });
          video.addEventListener("seeked", drawPassOverlay);
          video.addEventListener("play", () => { cancelAnimationFrame(passAnimFrame); paintPass(); });
          video.addEventListener("pause", () => cancelAnimationFrame(passAnimFrame));
          video.addEventListener("ended", () => cancelAnimationFrame(passAnimFrame));
          video.ontimeupdate = () => {
            if (video.currentTime >= clipEnd) { video.pause(); cancelAnimationFrame(passAnimFrame); }
          };
          panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
          });
        };
      });
    }
  }
  async function savePassReview(key, value) {
    const human = value === "yes" ? true : value === "no" ? false : null;
    try {
      state.results.passReview = await api(
        `/api/projects/${state.project.id}/reviews/passes/${encodeURIComponent(key)}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ human_is_pass: human, outcome: "", note: "" }),
        },
      );
      state.results.quality = await api(
        `/api/projects/${state.project.id}/quality`,
      );
      renderQuality();
      toast("复核结果已保存", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  }
  async function saveTeamLabels() {
    const labels = {};
    $$("[data-team-map]").forEach((el) => {
      const v = el.value.trim();
      if (v) labels[el.dataset.teamMap] = v;
    });
    try {
      await api(`/api/projects/${state.project.id}/meta`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ team_labels: labels }),
      });
      await loadProject(state.project.id);
      await loadResults();
      toast("队伍名称已更新到全部结果", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  }
  async function saveAssessment(gid, clear = false) {
    const scores = {};
    if (!clear) {
      assessmentDefs.forEach(([key]) => {
        const el = $(`[data-assessment-value="${key}"]`);
        if (el && el.value !== "") scores[key] = Number(el.value);
      });
      if (Object.keys(scores).length !== assessmentDefs.length) {
        toast(
          "正式八维评估需要填写全部 8 项；如暂不评估请保持待评估。",
          "error",
        );
        return;
      }
    }
    try {
      const selectedGid = gid;
      state.results.assessmentReview = await api(
        `/api/projects/${state.project.id}/reviews/assessments/${gid}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            scores: clear ? {} : scores,
            note: clear ? "" : $("#assessmentNote")?.value || "",
          }),
        },
      );
      state.results.players = await api(
        `/api/projects/${state.project.id}/players`,
      );
      const idx = state.results.players.findIndex((x) =>
        (x.global_ids || []).includes(selectedGid),
      );
      state.playerIndex = idx >= 0 ? idx : 0;
      renderPlayers();
      $("#reportFrame").src =
        `/api/projects/${state.project.id}/report?_=${Date.now()}`;
      toast(
        clear ? "八维评估已清除" : "八维评估已保存并进入正式球员卡",
        "success",
      );
    } catch (e) {
      toast(e.message, "error");
    }
  }
  async function saveIdentity(gid, clear = false) {
    const linked = [
      gid,
      ...$$("[data-link-gid]:checked:not(:disabled)").map((el) =>
        Number(el.dataset.linkGid),
      ),
    ];
    const payload = clear
      ? {
          name: "",
          jersey_number: "",
          team_id: "",
          roster_index: null,
          note: "",
          linked_global_ids: [gid],
        }
      : {
          name: $("#identityName")?.value || "",
          jersey_number: $("#identityNumber")?.value || "",
          team_id: $("#identityTeam")?.value || "",
          roster_index:
            $("#identityRoster")?.value !== "" && $("#identityRoster")
              ? Number($("#identityRoster").value)
              : null,
          note: $("#identityNote")?.value || "",
          linked_global_ids: [...new Set(linked)],
        };
    try {
      const selectedGid = gid;
      state.results.identityReview = await api(
        `/api/projects/${state.project.id}/reviews/identities/${gid}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        },
      );
      const [ps, ov, q] = await Promise.all([
        api(`/api/projects/${state.project.id}/players`),
        api(`/api/projects/${state.project.id}/overview`),
        api(`/api/projects/${state.project.id}/quality`),
      ]);
      state.results.players = ps;
      state.results.overview = ov;
      state.results.quality = q;
      const idx = ps.findIndex((x) =>
        (x.global_ids || []).includes(selectedGid),
      );
      state.playerIndex = idx >= 0 ? idx : 0;
      renderOverview();
      renderPlayers();
      renderQuality();
      toast(clear ? "身份确认已清除" : "身份确认已保存", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  }
  async function saveReportAnnotation(gid) {
    const fields = {};
    $$("[data-report-field]").forEach(
      (el) => (fields[el.dataset.reportField] = el.value || ""),
    );
    try {
      await api(
        `/api/projects/${state.project.id}/reviews/player-report/${gid}`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fields }),
        },
      );
      const base = `/api/projects/${state.project.id}/players/${state.playerIndex}/report`;
      $("#reportFrame").src = `${base}?_=${Date.now()}`;
      $("#openReportBtn").href = base;
      $("#downloadPlayerPdfBtn").href = `${base}.pdf`;
      toast("正式报告标注已保存", "success");
    } catch (e) {
      toast(e.message, "error");
    }
  }
  function renderFiles() {
    const rows = state.results.files || [];
    $("#fileList").innerHTML =
      rows
        .map(
          (f) =>
            `<div class="file-row"><div class="file-icon">${esc((f.filename || "FILE").split(".").pop().toUpperCase())}</div><div><b>${esc(f.label)}</b><span>${esc(f.category || "结果")} · ${esc(f.filename || "")}</span></div><span class="file-size">${fmtBytes(f.size_bytes)}</span><a class="btn ghost small" href="/api/projects/${state.project.id}/file/${f.index}">下载</a></div>`,
        )
        .join("") || '<div class="point-row">暂无可下载结果</div>';
  }
  async function switchResult(tab) {
    $$(".result-tab").forEach((x) =>
      x.classList.toggle("active", x.dataset.result === tab),
    );
    $$(".result-pane").forEach((x) =>
      x.classList.toggle("active", x.id === `result-${tab}`),
    );
    if (tab === "replay") await ensureReplay();
    if (tab === "players") renderPlayers();
    if (tab === "report") renderReportWorkflow();
  }

  async function loadSystem() {
    const s = await api("/api/system/status");
    state.system = s;
    $("#serverDot").classList.add("ok");
    $("#serverText").textContent = `V${s.version} · 服务正常`;
    const cards = [
      [
        "◈",
        "分析模型",
        s.model.ready ? "已就绪" : "未放置",
        s.model.ready
          ? fmtBytes(s.model.size_bytes)
          : "可在下方直接上传 yolov8x.pt",
      ],
      [
        "PKG",
        "分析依赖",
        s.readiness?.inference_dependencies ? "已就绪" : "缺少依赖",
        s.readiness?.inference_dependencies
          ? "Tracking / OCR / Metric 依赖完整"
          : "运行 Windows 安装脚本后重新检查",
      ],
      [
        "GPU",
        "计算资源",
        s.gpu.available ? "GPU 可用" : "CPU / 未检测 GPU",
        s.gpu.name || `Python ${s.python}`,
      ],
      [
        "WIN",
        s.platform?.windows ? "Windows 主机" : "当前构建主机",
        s.platform?.windows
          ? "Windows 运行中"
          : `${s.platform?.system || ""} · 发布包支持 Windows`,
        s.python_executable || "",
      ],
      [
        "FF",
        "视频工具",
        s.ffmpeg.ready ? "FFmpeg 可用" : "核心视频读写可用",
        s.ffmpeg.path || "核心链路使用 OpenCV；FFmpeg 为增强项",
      ],
      [
        "HD",
        "结果空间",
        `${fmtBytes(s.disk.free_bytes)} 可用`,
        s.disk.runtime_root,
      ],
    ];
    $("#systemCards").innerHTML = cards
      .map(
        (c, i) =>
          `<div class="system-card"><div class="system-icon">${c[0]}</div><h4>${esc(c[1])}</h4><p>${esc(c[3])}</p><span class="system-state ${(i === 0 && !s.model.ready) || (i === 1 && !s.readiness?.inference_dependencies) ? "warn" : ""}">${esc(c[2])}</span></div>`,
      )
      .join("");
    const labels = {
      tracking: "球员/足球追踪",
      onboarding: "视频适配 / 健康检查",
      metric: "米制跑动",
      dynamic_calibration: "多锚点动态标定",
      match_analysis: "球权 / 传球网络",
      jersey_ocr: "多帧号码识别",
      player_cards: "球员卡 / 报告",
      identity_audit: "身份质量审计",
    };
    $("#engineMatrix").innerHTML =
      Object.entries(s.engine || {})
        .map(
          ([k, ok]) =>
            `<div><b>${ok ? "✓" : "×"} ${esc(labels[k] || k)}</b><span>${ok ? "生产模块已纳入系统" : "模块缺失，请重新解压完整包"}</span></div>`,
        )
        .join("") +
      `<div><b>${s.chain_audit?.status === "passed" ? "✓" : "!"} 源码链路审计</b><span>${s.chain_audit?.critical_files_matched ?? "—"} / ${s.chain_audit?.critical_files_total ?? "—"} 个核心文件与原始仓库 SHA256 一致</span></div>`;
  }

  async function saveSettings() {
    const settings = {};
    $$("[data-setting]").forEach((el) => {
      const key = el.dataset.setting;
      let v = el.value;
      if (el.type === "number") v = Number(v);
      if (el.dataset.valueType === "choice" && (v === "true" || v === "false"))
        v = v === "true";
      settings[key] = v;
    });
    await api(`/api/projects/${state.project.id}/settings`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings }),
    });
    await loadProject(state.project.id);
    toast("分析参数已保存", "success");
  }
  async function saveMeta() {
    const payload = {};
    $$("[data-meta]").forEach((el) => (payload[el.dataset.meta] = el.value));
    await api(`/api/projects/${state.project.id}/meta`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    await loadProject(state.project.id);
    toast("比赛资料已保存", "success");
  }

  function bind() {
    $$(".nav-item[data-page]").forEach((b) =>
      b.addEventListener("click", () => switchPage(b.dataset.page)),
    );
    $("#projectSelect").addEventListener("change", (e) =>
      loadProject(e.target.value, true).catch((x) => toast(x.message, "error")),
    );
    $("#refreshBtn").addEventListener("click", () =>
      loadProjects(state.project?.id)
        .then(() => toast("已刷新", "success"))
        .catch((e) => toast(e.message, "error")),
    );
    $$("[data-project-filter]").forEach((b) =>
      b.addEventListener("click", () => {
        $$("[data-project-filter]").forEach((x) =>
          x.classList.toggle("active", x === b),
        );
        state.projectFilter = b.dataset.projectFilter;
        renderProjectCards();
      }),
    );
    const dlg = $("#newProjectDialog");
    const open = () => dlg.showModal();
    $("#heroNewBtn").onclick = open;
    $("#setupCreateBtn").onclick = open;
    $("#closeDialog").onclick = () => dlg.close();
    $("#cancelDialog").onclick = () => dlg.close();
    $("#newProjectForm").addEventListener("submit", async (e) => {
      e.preventDefault();
      const fd = new FormData();
      fd.append("name", $("#newProjectName").value.trim() || "新比赛");
      fd.append("home_team", $("#newHomeTeam").value.trim() || "主队");
      fd.append("away_team", $("#newAwayTeam").value.trim() || "客队");
      fd.append("age_group", $("#newAgeGroup").value.trim());
      try {
        const p = await api("/api/projects", { method: "POST", body: fd });
        dlg.close();
        $("#newProjectName").value = "";
        await loadProjects(p.id);
        switchPage("setup");
        toast("正式项目已创建，请上传比赛视频", "success");
      } catch (err) {
        toast(err.message, "error");
      }
    });
    $("#videoInput").addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("video", f);
      toast("正在上传并读取比赛视频…");
      try {
        await api(`/api/projects/${state.project.id}/video`, {
          method: "POST",
          body: fd,
        });
        await loadProject(state.project.id);
        toast("比赛视频上传完成", "success");
      } catch (err) {
        toast(err.message, "error");
      }
      e.target.value = "";
    });
    $("#rosterInput").addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("roster", f);
      try {
        await api(`/api/projects/${state.project.id}/roster`, {
          method: "POST",
          body: fd,
        });
        await loadProject(state.project.id);
        toast("球员名单已导入", "success");
      } catch (err) {
        toast(err.message, "error");
      }
      e.target.value = "";
    });
    $("#saveMetaBtn").onclick = () =>
      saveMeta().catch((e) => toast(e.message, "error"));
    $("#saveSettingsBtn").onclick = () =>
      saveSettings().catch((e) => toast(e.message, "error"));
    $("#settingsImportInput").addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("config", f);
      try {
        const out = await api(
          `/api/projects/${state.project.id}/settings/import`,
          { method: "POST", body: fd },
        );
        await loadProject(state.project.id);
        toast(`参数模板已导入 ${out.imported_keys?.length || 0} 项`, "success");
      } catch (err) {
        toast(err.message, "error");
      }
      e.target.value = "";
    });
    $$(".preset").forEach((b) =>
      b.addEventListener("click", () => {
        $$(".preset").forEach((x) => x.classList.toggle("active", x === b));
        const vals =
          b.dataset.preset === "quality"
            ? {
                confidence: 0.25,
                imgsz: 1280,
                event_count: 20,
                dynamic_sample_step: 5,
                identity_audit_enabled: true,
              }
            : b.dataset.preset === "balanced"
              ? {
                  confidence: 0.3,
                  imgsz: 960,
                  event_count: 16,
                  dynamic_sample_step: 7,
                  identity_audit_enabled: true,
                }
              : {
                  confidence: 0.35,
                  imgsz: 736,
                  event_count: 10,
                  dynamic_sample_step: 10,
                  identity_audit_enabled: false,
                };
        Object.entries(vals).forEach(([k, v]) => {
          const el = $(`[data-setting="${k}"]`);
          if (el) el.value = v;
        });
      }),
    );
    const selectCalibTab = (name) => {
      $$(".calib-tab").forEach((x) =>
        x.classList.toggle("active", x.dataset.calib === name),
      );
      $("#calibManual").classList.toggle("active", name === "manual");
      $("#calibUpload").classList.toggle("active", name === "upload");
    };
    $$(".calib-tab").forEach((b) =>
      b.addEventListener("click", () => selectCalibTab(b.dataset.calib)),
    );
    $("#calibFileInput").addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("config", f);
      try {
        state.calib.imported = null;
        state.calib.importedProjectId = null;
        await api(`/api/projects/${state.project.id}/calibration/upload`, {
          method: "POST",
          body: fd,
        });
        await loadProject(state.project.id);
        selectCalibTab("manual");
        toast("标定已恢复到参考帧，可直接核对点位与验证线段", "success");
      } catch (err) {
        toast(err.message, "error");
      }
      e.target.value = "";
    });
    $("#loadFrameBtn").addEventListener("click", () => {
      if (!state.project?.video) {
        toast("请先上传比赛视频", "error");
        return;
      }
      const fi = Math.max(
        0,
        Math.min(
          Number($("#calibFrame").value) || 0,
          state.project.video.frame_count - 1,
        ),
      );
      if (
        state.calib.imported &&
        fi !== Number(state.calib.imported.frame_index)
      )
        state.calib.imported = null;
      updateCalibLists();
      loadCalibrationFrame(fi, !!state.calib.imported);
    });
    $("#armFitPoint").onclick = () => {
      state.calib.imported = null;
      updateCalibLists();
      state.calib.clickMode = "fit";
      $("#calibHint").textContent =
        `请点击球场坐标 (${Number($("#worldX").value)}, ${Number($("#worldY").value)}) 在画面中的位置。`;
      drawCalibOverlay();
    };
    $("#armValidation").onclick = () => {
      state.calib.imported = null;
      updateCalibLists();
      state.calib.clickMode = "validation";
      state.calib.validationDraft = [];
      $("#calibHint").textContent = "请依次点击独立验证线段的两个端点。";
      drawCalibOverlay();
    };
    $("#calibImage").addEventListener("click", onCalibImageClick);
    window.addEventListener("resize", drawCalibOverlay);
    $("#resetPointsBtn").onclick = clearAnchorDraft;
    $("#saveAnchorBtn").onclick = saveAnchor;
    $("#buildDynamicBtn").addEventListener("click", async () => {
      try {
        $("#buildDynamicBtn").disabled = true;
        await api(`/api/projects/${state.project.id}/calibration/expand`, {
          method: "POST",
        });
        await loadProject(state.project.id);
        toast("多视角逐帧动态标定已开始", "success");
      } catch (e) {
        toast(e.message, "error");
        await loadProject(state.project.id);
      }
    });
    const timelineSlider = $("#calibTimelineSlider"),
      timelineWrap = $("#calibTimeline .timeline-filmstrip-wrap"),
      scrubVideo = $("#calibScrubVideo"),
      calibImage = $("#calibImage"),
      frameAtPointer = (e) => {
        const rect = timelineWrap.getBoundingClientRect(),
          ratio = Math.max(
            0,
            Math.min(1, (e.clientX - rect.left) / Math.max(1, rect.width)),
          ),
          total = Math.max(1, Number(state.project?.video?.frame_count) || 1);
        return Math.round(ratio * (total - 1));
      };
    $("#calibTimelinePlay").onclick = () => {
      if (state.calib.playing) {
        const fi = setCalibFrameUi(
          Math.round(
            scrubVideo.currentTime * (Number(state.project?.video?.fps) || 30),
          ),
        );
        setCalibPlaying(false);
        loadCalibrationFrame(fi);
      } else setCalibPlaying(true);
    };
    timelineSlider.oninput = (e) =>
      previewCalibrationFrame(Number(e.target.value));
    timelineSlider.onchange = (e) =>
      loadCalibrationFrame(Number(e.target.value));
    timelineWrap.addEventListener("pointerdown", (e) => {
      if (!state.project?.video) return;
      e.preventDefault();
      setCalibPlaying(false);
      state.calib.dragPointerId = e.pointerId;
      timelineWrap.setPointerCapture(e.pointerId);
      previewCalibrationFrame(frameAtPointer(e));
    });
    timelineWrap.addEventListener("pointermove", (e) => {
      if (state.calib.dragPointerId !== e.pointerId) return;
      previewCalibrationFrame(frameAtPointer(e));
    });
    const finishTimelineDrag = (e) => {
      if (state.calib.dragPointerId !== e.pointerId) return;
      const fi = frameAtPointer(e);
      state.calib.dragPointerId = null;
      if (timelineWrap.hasPointerCapture(e.pointerId))
        timelineWrap.releasePointerCapture(e.pointerId);
      loadCalibrationFrame(fi);
    };
    timelineWrap.addEventListener("pointerup", finishTimelineDrag);
    timelineWrap.addEventListener("pointercancel", finishTimelineDrag);
    $("#calibTimelineStrip").onclick = (e) => {
      const item = e.target.closest("[data-timeline-frame]");
      if (item) {
        setCalibPlaying(false);
        loadCalibrationFrame(Number(item.dataset.timelineFrame));
      }
    };
    $$("[data-calib-step]").forEach(
      (b) =>
        (b.onclick = () => {
          setCalibPlaying(false);
          loadCalibrationFrame(
            Number($("#calibFrame").value) + Number(b.dataset.calibStep),
          );
        }),
    );
    scrubVideo.addEventListener("loadedmetadata", seekCalibScrubVideo);
    scrubVideo.addEventListener("seeked", () => {
      if (!state.calib.playing) return;
      scrubVideo.classList.add("active");
      calibImage.classList.add("scrub-hidden");
      drawCalibOverlay();
    });
    $("#runPreflightBtn").onclick = () =>
      refreshPreflight()
        .then(() => toast("启动检查已刷新", "success"))
        .catch((e) => toast(e.message, "error"));
    $("#startRunBtn").addEventListener("click", async () => {
      try {
        await api(`/api/projects/${state.project.id}/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ from_step: "tracking" }),
        });
        await loadProject(state.project.id);
        switchPage("progress");
        toast("正式分析已启动", "success");
      } catch (e) {
        toast(e.message, "error");
      }
    });
    $("#cancelBtn").onclick = async () => {
      await api(`/api/projects/${state.project.id}/cancel`, { method: "POST" });
      toast("已请求取消任务");
    };
    $("#retryBtn").onclick = async () => {
      const pipe = state.project.pipeline || {};
      const from = pipe.current_step || "tracking";
      if (!["failed", "interrupted", "cancelled"].includes(pipe.state)) {
        toast("当前状态不允许重试", "error");
        return;
      }
      try {
        await api(`/api/projects/${state.project.id}/run`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ from_step: from }),
        });
        await loadProject(state.project.id);
        toast(`已从 ${{ tracking: "追踪", jersey: "号码识别", events: "事件检测", report: "报告生成" }[from] || from} 步骤重新启动`, "success");
      } catch (e) {
        toast(e.message, "error");
      }
    };
    $$(".result-tab").forEach((b) =>
      b.addEventListener("click", () =>
        switchResult(b.dataset.result).catch((e) => toast(e.message, "error")),
      ),
    );
    $$("[data-result-jump]").forEach((b) =>
      b.addEventListener("click", () =>
        switchResult(b.dataset.resultJump).catch((e) =>
          toast(e.message, "error"),
        ),
      ),
    );
    $$("[data-go-setup]").forEach(
      (b) => (b.onclick = () => switchPage("setup")),
    );
    $$("[data-leader]").forEach((b) =>
      b.addEventListener("click", () => {
        $$("[data-leader]").forEach((x) =>
          x.classList.toggle("active", x === b),
        );
        state.leaderMode = b.dataset.leader;
        renderLeaders();
      }),
    );
    $("#eventTypeFilter").onchange = renderEvents;
    $("#eventSearch").oninput = renderEvents;
    $("#playerSearch").oninput = renderPlayers;
    $("#replayPlay").onclick = () => {
      ensureReplay().catch(() => {});
      setReplayPlaying(!state.replay.playing);
    };
    $("#replaySlider").oninput = async (e) => {
      setReplayPlaying(false);
      const sourceFrame = Number(e.target.value),
        fps = Number(state.replay.data?.fps || 30),
        time = sourceFrame / fps;
      await loadReplayWindowForFrame(sourceFrame);
      state.replay.index = replayFrameAtTime(time);
      state.replay.pitchTime = time;
      $("#analysisVideo").currentTime = time;
      drawReplay();
    };
    $("#replaySpeed").onchange = (e) => {
      state.replay.speed = Number(e.target.value) || 1;
      $("#analysisVideo").playbackRate = state.replay.speed;
      if (state.replay.playing) {
        setReplayPlaying(false);
        setReplayPlaying(true);
      }
    };
    $$("[data-replay-mode]").forEach(
      (b) => (b.onclick = () => setReplayMode(b.dataset.replayMode)),
    );
    const setupFullscreen = (btnId, targetSelector) => {
      const btn = $(btnId);
      if (!btn) return;
      btn.onclick = async () => {
        const target = $(targetSelector);
        if (!target) return;
        if (document.fullscreenElement === target) {
          await document.exitFullscreen();
          btn.textContent = "⛶ 放大";
        } else {
          try {
            await target.requestFullscreen();
            btn.textContent = "⛶ 还原";
          } catch {}
        }
      };
    };
    setupFullscreen("#liveFullscreen", "#liveReplayStage");
    setupFullscreen("#pitchFullscreen", "#pitchCanvasWrapper");
    $("#analysisVideo").addEventListener("seeked", drawLiveReplay);
    $("#printReportBtn").onclick = () => {
      try {
        $("#reportFrame").contentWindow.print();
      } catch {
        window.open(`/api/projects/${state.project.id}/report`, "_blank");
      }
    };
    $("#refreshSystemBtn").onclick = () =>
      loadSystem()
        .then(() => toast("系统状态已刷新", "success"))
        .catch((e) => toast(e.message, "error"));
    $("#modelInput").addEventListener("change", async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      const fd = new FormData();
      fd.append("model", f);
      $("#modelInstallState").textContent = "正在安装模型，请勿关闭页面…";
      try {
        await api("/api/system/model", { method: "POST", body: fd });
        $("#modelInstallState").textContent = "模型安装完成。";
        await loadSystem();
        toast("分析模型已安装", "success");
      } catch (err) {
        $("#modelInstallState").textContent = `模型安装失败：${err.message}`;
        toast(err.message, "error");
      }
      e.target.value = "";
    });
  }
  async function init() {
    bind();
    updateCalibLists();
    try {
      await loadSystem();
      await loadProjects();
      switchPage("projects");
    } catch (e) {
      toast(`系统初始化失败：${e.message}`, "error");
    }
  }
  init();
})();
