(function () {
  "use strict";
  const C = window.CardScope;
  const $ = (id) => document.getElementById(id);
  const state = {
    items: [], selected: null, correctedInner: null, correctedOuter: null,
    getInnerCorrection: null, getOuterCorrection: null, rectifiedObjectUrl: null,
    tenants: [], admins: [], inspections: [], selectedInspection: null, training: null, trainingLoading: false,
    session: null,
    view: "feedback", feedbackLoading: false, feedbackReloadQueued: false, inspectionLoading: false, reviewInFlight: false,
    outerZoom: null, innerZoom: null, annotationMode: false, rectifiedOuterKey: "",
    recordOuterZoom: null, recordInnerZoom: null,
    sourceSize: { width: 1, height: 1 },
  };
  const TAG_NAMES = {
    outer_frame_wrong: "外框错误", inner_frame_wrong: "内框错误", inner_left_wrong: "左内框错误", inner_right_wrong: "右内框错误",
    inner_top_wrong: "上内框错误", inner_bottom_wrong: "下内框错误", glare_or_reflection: "反光干扰",
    shadow_interference: "阴影干扰", perspective_extreme: "透视过大", card_cropped: "卡牌裁切", other: "其他",
  };
  const STATUS_NAMES = { pending: "待审核", approved: "已批准", needs_annotation: "需高级标注", discarded: "已舍弃", rejected: "已驳回" };
  const INSPECTION_STATE_NAMES = {
    completed: "待企业确认", confirmed: "企业已确认", detection_failed: "检测失败",
    feedback_pending: "反馈待审核", feedback_approved: "已批准入池",
    feedback_needs_annotation: "需高级标注", feedback_discarded: "样本已舍弃", feedback_rejected: "反馈已驳回",
  };

  async function init() {
    bindEvents();
    setupAnnotationTools();
    setupInspectionRecordTools();
    try {
      const payload = await C.api("/session");
      if (payload.session.role !== "admin") throw new Error("此链接不是内部管理链接。");
      state.session = payload.session;
      $("adminName").textContent = payload.session.display_name;
      applySessionPermissions();
      await loadFeedback();
      window.setInterval(() => {
        if (state.view === "feedback" && !document.hidden) loadFeedback({ silent: true });
        if (state.view === "inspections" && !document.hidden) loadInspections({ silent: true });
        if (state.view === "training" && !document.hidden) loadTraining({ silent: true });
      }, 8000);
    } catch (error) {
      if (!C.token()) {
        window.location.replace("/admin-login");
        return;
      }
      C.toast(error.message, "error");
      $("feedbackList").innerHTML = '<div class="empty-list">管理链接无效</div>';
    }
  }

  function bindEvents() {
    for (const button of document.querySelectorAll("[data-admin-view]")) {
      button.addEventListener("click", () => switchView(button.dataset.adminView));
    }
    $("statusFilter").addEventListener("change", loadFeedback);
    $("approveFeedback").addEventListener("click", () => review("approve"));
    $("reopenFeedback").addEventListener("click", () => review("reopen"));
    $("needsAnnotation").addEventListener("click", () => review("needs_annotation"));
    $("rejectFeedback").addEventListener("click", () => review("reject"));
    $("discardFeedback").addEventListener("click", () => review("discard"));
    $("deleteFeedback").addEventListener("click", deleteSelectedFeedback);
    $("exportBundle").addEventListener("click", exportBundle);
    $("exportTraining").addEventListener("click", exportTraining);
    $("refreshRectification").addEventListener("click", refreshRectification);
    $("toggleAnnotationMode").addEventListener("click", toggleAnnotationMode);
    $("resetAnnotations").addEventListener("click", resetAnnotations);
    $("refreshFeedback").addEventListener("click", () => loadFeedback());
    $("refreshInspections").addEventListener("click", () => loadInspections());
    $("refreshTraining").addEventListener("click", () => loadTraining());
    $("saveTrainingSettings").addEventListener("click", saveTrainingSettings);
    $("startTraining").addEventListener("click", startTraining);
    $("rollbackTrainingModel").addEventListener("click", rollbackTrainingModel);
    $("inspectionTenantFilter").addEventListener("change", renderInspectionList);
    $("inspectionStateFilter").addEventListener("change", renderInspectionList);
    $("inspectionSearch").addEventListener("input", renderInspectionList);
    $("openInspectionFeedback").addEventListener("click", openSelectedInspectionFeedback);
    $("createTenantButton").addEventListener("click", () => $("tenantDialog").showModal());
    $("closeTenantDialog").addEventListener("click", closeTenantDialog);
    $("cancelTenant").addEventListener("click", closeTenantDialog);
    $("tenantForm").addEventListener("submit", createTenant);
    $("closeCredentialDialog").addEventListener("click", closeCredentialDialog);
    $("dismissCredentialDialog").addEventListener("click", closeCredentialDialog);
    $("copyCredentialButton").addEventListener("click", async () => {
      try {
        await copyText($("credentialText").value);
        C.toast("企业登录信息已复制。");
      } catch (error) { C.toast(error.message, "error"); }
    });
    $("createAdminButton").addEventListener("click", () => $("adminUserDialog").showModal());
    $("closeAdminUserDialog").addEventListener("click", closeAdminUserDialog);
    $("cancelAdminUser").addEventListener("click", closeAdminUserDialog);
    $("adminUserForm").addEventListener("submit", createAdminUser);
    $("adminLogout").addEventListener("click", logoutAdmin);
    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.annotationMode) toggleAnnotationMode(false);
    });
  }

  function isOwner() {
    return Boolean(state.session?.permissions?.manage_admins);
  }

  function applySessionPermissions() {
    const owner = isOwner();
    $("adminRoleLabel").textContent = owner ? "平台主管理员" : "标注管理员 · 仅人工标注";
    $("adminLogout").classList.toggle("hidden", state.session?.auth_type !== "password");
    for (const node of document.querySelectorAll("[data-owner-only]")) {
      node.classList.toggle("hidden", !owner);
    }
    for (const node of document.querySelectorAll("[data-annotator-only]")) {
      node.classList.toggle("hidden", owner);
    }
    $("statusFilter").disabled = !owner;
    if (!owner) $("statusFilter").value = "pending";
    $("approveFeedback").textContent = owner ? "审核通过并加入训练池" : "提交人工标注并加入训练池";
  }

  async function logoutAdmin() {
    try { await C.api("/auth/logout", { method: "POST", json: {} }); } catch (_) { /* clear locally */ }
    C.clearToken();
    window.location.replace("/admin-login");
  }

  function setupAnnotationTools() {
    state.outerZoom = C.zoomPanControls($("adminOuterStage"), $("adminOuterSurface"), { width: 1, height: 1 }, {
      fit: $("outerFit"), zoomOut: $("outerZoomOut"), zoomIn: $("outerZoomIn"), zoomText: $("outerZoomText"),
    });
    state.innerZoom = C.zoomPanControls($("adminInnerStage"), $("adminInnerSurface"), { width: 630, height: 880 }, {
      fit: $("innerFit"), zoomOut: $("innerZoomOut"), zoomIn: $("innerZoomIn"), zoomText: $("innerZoomText"),
    });
    $("adminOuterImage").addEventListener("load", () => state.outerZoom?.fit());
    $("adminInnerImage").addEventListener("load", () => state.innerZoom?.fit());
    $("adminOuterImage").addEventListener("error", () => setActionStatus("外框原图加载失败，请刷新页面后重试。", "error"));
    $("adminInnerImage").addEventListener("error", () => setActionStatus("内框校正图加载失败，请刷新页面后重试。", "error"));
  }

  function setupInspectionRecordTools() {
    state.recordOuterZoom = C.zoomPanControls($("recordOuterViewport"), $("recordOuterSurface"), { width: 1, height: 1 }, {
      fit: $("recordOuterFit"), zoomOut: $("recordOuterZoomOut"), zoomIn: $("recordOuterZoomIn"),
      zoomText: $("recordOuterZoomText"), panWithPrimary: true,
    });
    state.recordInnerZoom = C.zoomPanControls($("recordInnerViewport"), $("recordInnerSurface"), { width: 630, height: 880 }, {
      fit: $("recordInnerFit"), zoomOut: $("recordInnerZoomOut"), zoomIn: $("recordInnerZoomIn"),
      zoomText: $("recordInnerZoomText"), panWithPrimary: true, lineScreenPx: 0.7, lineHaloScreenPx: 0.15,
    });
    $("recordOuterImage").addEventListener("load", () => state.recordOuterZoom?.fit());
    $("recordInnerImage").addEventListener("load", () => state.recordInnerZoom?.fit());
  }

  function toggleAnnotationMode(force) {
    state.annotationMode = typeof force === "boolean" ? force : !state.annotationMode;
    $("reviewPanel").classList.toggle("annotation-maximized", state.annotationMode);
    document.body.classList.toggle("annotation-mode", state.annotationMode);
    $("toggleAnnotationMode").textContent = state.annotationMode ? "退出全屏标注" : "全屏精细标注";
    window.requestAnimationFrame(() => { state.outerZoom?.fit(); state.innerZoom?.fit(); });
  }

  function outerKey(points) {
    return Array.isArray(points) ? points.map((point) => point.map((value) => Number(value).toFixed(1)).join(",")).join(";") : "";
  }

  function outerDraftProblem(points) {
    if (!Array.isArray(points) || points.length !== 4) return "请确认左上、右上、右下、左下四个绿色角点。";
    const width = Number(state.sourceSize?.width || 1);
    const height = Number(state.sourceSize?.height || 1);
    if (points.some((point) => !Array.isArray(point) || point.length !== 2 || point.some((value) => !Number.isFinite(Number(value))))) return "外框角点包含无效坐标。";
    if (points.some(([x, y]) => x < 0 || x > width - 1 || y < 0 || y > height - 1)) return "外框角点超出了原图范围。";
    let signedArea = 0;
    for (let index = 0; index < 4; index += 1) {
      const next = points[(index + 1) % 4];
      signedArea += points[index][0] * next[1] - next[0] * points[index][1];
    }
    if (signedArea <= 0) return "外框方向错误，请按左上、右上、右下、左下顺序调整。";
    let sign = 0;
    for (let index = 0; index < 4; index += 1) {
      const a = points[index], b = points[(index + 1) % 4], c = points[(index + 2) % 4];
      const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
      if (Math.abs(cross) < 0.001) return "外框存在共线角点。";
      const current = Math.sign(cross);
      if (sign && current !== sign) return "外框发生交叉，请检查四角顺序。";
      sign = current;
    }
    if (Math.abs(signedArea) / 2 < width * height * 0.01) return "外框面积过小，请重新标注。";
    return "";
  }

  function resetAnnotations() {
    if (!state.selected || state.selected.review_status !== "pending" || state.reviewInFlight) return;
    const item = state.selected;
    const notes = $("adminReviewNotes").value;
    item.corrected_inner = null;
    item.corrected_outer = null;
    selectFeedback(item);
    $("adminReviewNotes").value = notes;
    setActionStatus("已恢复这张图片的模型检测结果；可以重新拖动绿点和红线。", "");
  }

  function setActionStatus(message = "", type = "") {
    const node = $("reviewActionStatus");
    node.textContent = message;
    node.className = `review-action-status${message ? "" : " hidden"}${type ? ` ${type}` : ""}`;
  }

  function showAdminNotice(message, type = "") {
    const node = $("adminNotice");
    node.textContent = message;
    node.className = `admin-notice${message ? "" : " hidden"}${type ? ` ${type}` : ""}`;
  }

  function setReviewControlsDisabled(disabled) {
    for (const id of ["approveFeedback", "needsAnnotation", "rejectFeedback", "discardFeedback", "reopenFeedback", "refreshRectification", "resetAnnotations"]) $(id).disabled = disabled;
    for (const input of document.querySelectorAll("#adminOuterControls input, #adminCorrectionControls input")) input.disabled = disabled;
    $("adminOuterStage").dataset.locked = String(disabled);
    $("adminInnerStage").dataset.locked = String(disabled);
  }

  async function switchView(view) {
    const allowed = isOwner() ? ["feedback", "inspections", "training", "tenants", "admins"] : ["feedback"];
    state.view = allowed.includes(view) ? view : "feedback";
    const feedback = state.view === "feedback";
    const inspections = state.view === "inspections";
    const training = state.view === "training";
    const tenants = state.view === "tenants";
    const admins = state.view === "admins";
    $("feedbackAdminView").classList.toggle("hidden", !feedback);
    $("inspectionAdminView").classList.toggle("hidden", !inspections);
    $("trainingAdminView").classList.toggle("hidden", !training);
    $("tenantAdminView").classList.toggle("hidden", !tenants);
    $("adminUsersView").classList.toggle("hidden", !admins);
    $("exportBundle").classList.toggle("hidden", !feedback || !isOwner());
    $("exportTraining").classList.toggle("hidden", !feedback || !isOwner());
    $("historyExportOption").classList.toggle("hidden", !feedback || !isOwner());
    $("refreshFeedback").classList.toggle("hidden", !feedback);
    $("refreshInspections").classList.toggle("hidden", !inspections);
    $("refreshTraining").classList.toggle("hidden", !training);
    $("createTenantButton").classList.toggle("hidden", !tenants || !isOwner());
    $("createAdminButton").classList.toggle("hidden", !admins || !isOwner());
    const titles = {
      feedback: ["模型反馈审核台", "同时修正外框四角和内框四线，审核后可批量导出训练数据"],
      inspections: ["企业检测记录", "企业每完成一张检测都会自动保存在这里，可以查看内外框、居中度和反馈状态"],
      training: ["模型自动训练与安全更新", "用审核后的实拍标注建立候选模型，自动分析误差并在通过质量门禁后更新检测页面"],
      tenants: ["企业访问管理", "为每家企业创建独立检测空间，管理链接有效期和使用状态"],
      admins: ["管理员账号管理", "创建独立标注账号，并控制启用状态和登录密码"],
    };
    $("adminViewTitle").textContent = titles[state.view][0];
    $("adminViewSubtitle").textContent = titles[state.view][1];
    for (const button of document.querySelectorAll("[data-admin-view]")) {
      button.classList.toggle("selected", button.dataset.adminView === state.view);
    }
    if (feedback) await loadFeedback({ force: true });
    else if (inspections) await loadInspections();
    else if (training) await loadTraining();
    else if (tenants) await loadTenants();
    else await loadAdminUsers();
  }

  const TRAINING_STATUS_NAMES = {
    queued: "排队中", preparing: "整理数据", training: "训练候选模型", evaluating: "实拍对比评测",
    promoting: "安全更新模型", completed: "已完成", failed: "失败", blocked: "已阻止",
  };
  const TRAINING_PROGRESS = { queued: 4, preparing: 16, training: 58, evaluating: 82, promoting: 94, completed: 100, failed: 100, blocked: 100 };

  function setTrainingNotice(message = "", type = "") {
    const node = $("trainingNotice");
    node.textContent = message;
    node.className = `admin-notice${message ? "" : " hidden"}${type ? ` ${type}` : ""}`;
  }

  async function loadTraining({ silent = false } = {}) {
    if (state.trainingLoading) return;
    state.trainingLoading = true;
    $("refreshTraining").disabled = true;
    try {
      const payload = await C.api("/admin/training");
      state.training = payload;
      renderTraining(payload);
      if (!silent) C.toast("自动训练状态已刷新。");
    } catch (error) {
      setTrainingNotice(`读取训练状态失败：${error.message}`, "error");
      if (!silent) C.toast(error.message, "error");
    } finally {
      state.trainingLoading = false;
      $("refreshTraining").disabled = false;
    }
  }

  function renderTraining(payload) {
    const settings = payload.settings || {};
    const readiness = payload.readiness || {};
    const gpu = readiness.gpu || {};
    const activeModel = payload.active_model || {};
    $("trainingApprovedCount").textContent = readiness.approved_samples ?? 0;
    $("trainingNewCount").textContent = `${readiness.new_samples_since_last_job ?? 0} / ${readiness.minimum_new_samples ?? 0}`;
    $("trainingGpuState").textContent = gpu.available ? (gpu.name || "CUDA 可用") : "CUDA 不可用";
    $("trainingGpuState").title = gpu.available && gpu.memory_gb ? `${gpu.memory_gb} GB 显存` : (gpu.error || "需要 NVIDIA CUDA 显卡");
    $("trainingModelVersion").textContent = activeModel.package_version || "unknown";
    const editingSettings = Boolean(document.activeElement?.closest?.(".training-settings-card"));
    if (!editingSettings) {
      $("trainingEnabled").checked = Boolean(settings.enabled);
      $("trainingAutoPromote").checked = Boolean(settings.auto_promote);
      $("trainingOfflineOptimization").checked = Boolean(settings.offline_optimization);
      $("trainingHardReplay").checked = Boolean(settings.hard_example_replay);
      $("trainingMinimumApproved").value = settings.minimum_approved_samples ?? 20;
      $("trainingMinimumNew").value = settings.minimum_new_samples ?? 10;
      $("trainingEpochs").value = settings.epochs ?? 25;
      $("trainingHistoryLimit").value = settings.history_limit ?? 100;
      $("trainingOptimizationTrials").value = settings.optimization_trials ?? 2;
      $("trainingScreeningEpochs").value = settings.screening_epochs ?? 6;
      for (const input of document.querySelectorAll("[data-training-target]")) input.checked = (settings.targets || []).includes(input.dataset.trainingTarget);
    }
    const automationBadge = $("trainingAutomationBadge");
    automationBadge.textContent = settings.enabled ? "自动训练已启用" : "自动训练未启用";
    automationBadge.className = `training-pill ${settings.enabled ? "pass" : "neutral"}`;
    const active = payload.active_job;
    const readyMessage = !gpu.available
      ? "当前没有可用 CUDA 显卡，只能检测和审核，不能启动深度学习训练。"
      : (readiness.approved_samples || 0) < (readiness.minimum_approved_samples || 20)
        ? `还需批准 ${(readiness.minimum_approved_samples || 20) - (readiness.approved_samples || 0)} 张实拍标注，才达到最低训练门槛。`
        : active ? `任务 ${active.id} 正在${TRAINING_STATUS_NAMES[active.status] || active.status}。`
          : "样本和训练设备均已就绪，可以建立候选模型。";
    $("trainingReadiness").textContent = readyMessage;
    $("startTraining").disabled = !readiness.ready || Boolean(active);
    renderTrainingJob(active, payload.jobs || []);
    renderTrainingAnalysis(payload);
  }

  function renderTrainingJob(active, jobs) {
    const current = active || jobs[0] || null;
    const badge = $("trainingJobBadge");
    const progress = current ? (TRAINING_PROGRESS[current.status] ?? 0) : 0;
    $("trainingProgress").querySelector("span").style.width = `${progress}%`;
    badge.textContent = current ? (TRAINING_STATUS_NAMES[current.status] || current.status) : "空闲";
    badge.className = `training-pill ${current?.status === "completed" ? "pass" : ["failed", "blocked"].includes(current?.status) ? "fail" : active ? "running" : "neutral"}`;
    const currentNode = $("trainingJobCurrent");
    if (!current) {
      currentNode.className = "training-current empty";
      currentNode.innerHTML = "<strong>尚未启动训练任务</strong><span>审核后的样本达到门槛即可手动或自动开始。</span>";
    } else {
      const targets = (current.targets || []).map(trainingTargetName).join("、") || "—";
      const activeTarget = Object.entries(current.target_statuses || {}).find(([, value]) => ["screening", "training", "training_best"].includes(value?.status));
      const optimizerProgress = activeTarget
        ? `；当前 ${trainingTargetName(activeTarget[0])}：${activeTarget[1].status === "screening" ? `筛选 ${activeTarget[1].trial || 1}/${activeTarget[1].trial_count || 1}` : "完整训练最佳方案"}${activeTarget[1].profile ? `（${activeTarget[1].profile}）` : ""}`
        : "";
      currentNode.className = `training-current ${current.status || ""}`;
      currentNode.innerHTML = `<strong>${escapeHtml(TRAINING_STATUS_NAMES[current.status] || current.status)} · ${escapeHtml(current.id)}</strong><span>训练目标：${escapeHtml(targets)}；批准样本快照 ${Number(current.approved_snapshot_count || 0)} 张${escapeHtml(optimizerProgress)}；${escapeHtml(formatTrainingTime(current.updated_at))}</span>${current.error ? `<small>${escapeHtml(current.error)}</small>` : ""}`;
    }
    const list = $("trainingJobList"); list.replaceChildren();
    if (!jobs.length) { list.innerHTML = '<div class="training-job-empty">暂无历史训练任务</div>'; return; }
    for (const job of jobs.slice(0, 8)) {
      const item = document.createElement("div");
      const gate = job.quality_gate;
      const result = job.status === "completed" ? (gate?.passed ? "通过质量门禁" : "候选未达标，保留旧模型") : (TRAINING_STATUS_NAMES[job.status] || job.status);
      item.className = "training-job-row";
      item.innerHTML = `<div><strong>${escapeHtml(job.id)}</strong><span>${escapeHtml(formatTrainingTime(job.created_at))}</span></div><div><span>${escapeHtml(result)}</span><small>${Number(job.approved_snapshot_count || 0)} 张</small></div>`;
      list.append(item);
    }
  }

  function renderTrainingAnalysis(payload) {
    const jobs = payload.jobs || [];
    const job = jobs.find((row) => row.analysis || row.quality_gate) || null;
    const gate = job?.quality_gate || null;
    const badge = $("trainingGateBadge");
    badge.textContent = !gate ? "暂无报告" : gate.passed ? "质量门禁通过" : "候选模型未达标";
    badge.className = `training-pill ${!gate ? "neutral" : gate.passed ? "pass" : "fail"}`;
    const baseline = job?.analysis?.baseline || {};
    const candidate = job?.analysis?.candidate || {};
    const metricCards = $("trainingMetricGrid").children;
    setMetricCard(metricCards[0], candidate.outer_corner_mean_percent_diagonal, baseline.outer_corner_mean_percent_diagonal, "% 对角线");
    setMetricCard(metricCards[1], candidate.inner_edge_mae_px, baseline.inner_edge_mae_px, " px");
    metricCards[2].querySelector("strong").textContent = gate ? `${gate.holdout_count || 0} 张` : "—";
    metricCards[2].querySelector("small").textContent = gate?.holdout_count >= 5 ? "独立实拍留出集" : "至少需要 5 张固定留出图";
    const recommendations = $("trainingRecommendations"); recommendations.replaceChildren();
    const notes = job?.analysis?.recommendations || gate?.reasons || ["完成第一轮候选模型评测后，这里会显示反光、暗部、模糊、透视和边线方向性偏差。"];
    for (const note of notes) { const li = document.createElement("li"); li.textContent = note; recommendations.append(li); }
    const optimizerJob = payload.active_job?.offline_optimization_report ? payload.active_job : job;
    renderOfflineOptimization(optimizerJob?.analysis?.offline_optimization || optimizerJob?.offline_optimization_report || null);
    const deployment = payload.active_deployment;
    const dl = $("trainingDeployment");
    dl.innerHTML = `<div><dt>状态</dt><dd>${deployment ? "自动模型已上线" : "使用现有模型"}</dd></div><div><dt>最近更新</dt><dd>${escapeHtml(deployment ? formatTrainingTime(deployment.created_at) : "—")}</dd></div><div><dt>可回滚版本</dt><dd>${escapeHtml(deployment?.id || "—")}</dd></div>`;
    $("rollbackTrainingModel").disabled = !deployment || Boolean(payload.active_job);
  }

  function renderOfflineOptimization(report) {
    const mode = $("trainingOptimizerMode");
    const targets = $("trainingOptimizerTargets");
    targets.replaceChildren();
    if (!report) {
      mode.textContent = "等待首次运行";
      targets.innerHTML = "<span>运行后将显示每个模型尝试过的方案、验证指标和最终选择。</span>";
      return;
    }
    mode.textContent = report.enabled ? "本地参数搜索 · 未使用 API" : "固定单候选训练";
    const entries = Object.entries(report.targets || {});
    if (!entries.length) {
      targets.innerHTML = "<span>正在分析数据并建立本地筛选方案…</span>";
      return;
    }
    for (const [target, value] of entries) {
      const trials = value.trials || [];
      const selected = value.selected;
      const node = document.createElement("div");
      node.className = "offline-optimizer-target";
      const metric = selected?.validation?.available
        ? `${Number(selected.validation.value).toFixed(4)} · ${selected.validation.direction === "minimize" ? "越低越好" : "越高越好"}`
        : "等待验证指标";
      node.innerHTML = `<b>${escapeHtml(trainingTargetName(target))}</b><span>已筛选 ${trials.length} 个方案</span><small>${selected ? `选中：${escapeHtml(selected.label)}；${escapeHtml(metric)}` : "正在筛选…"}</small>`;
      targets.append(node);
    }
  }

  function setMetricCard(node, after, before, unit) {
    node.classList.remove("metric-improved", "metric-worse");
    const hasAfter = Number.isFinite(Number(after));
    const hasBefore = Number.isFinite(Number(before));
    node.querySelector("strong").textContent = hasAfter ? `${Number(after).toFixed(3)}${unit}` : "—";
    if (!hasAfter || !hasBefore) node.querySelector("small").textContent = "等待当前/候选成对评测";
    else {
      const delta = Number(after) - Number(before);
      node.querySelector("small").textContent = `当前模型 ${Number(before).toFixed(3)}${unit}；候选${delta <= 0 ? "改善" : "恶化"} ${Math.abs(delta).toFixed(3)}${unit}`;
      node.classList.toggle("metric-improved", delta <= 0);
      node.classList.toggle("metric-worse", delta > 0);
    }
  }

  function trainingTargetName(value) { return { outer_seg: "外框", inner_seg: "内框", inner_refiner: "内框精修" }[value] || value; }
  function formatTrainingTime(value) { return value ? C.formatDate(value) : "时间未知"; }

  function trainingSettingsPayload() {
    return {
      enabled: $("trainingEnabled").checked,
      minimum_approved_samples: Number($("trainingMinimumApproved").value),
      minimum_new_samples: Number($("trainingMinimumNew").value),
      epochs: Number($("trainingEpochs").value),
      history_limit: Number($("trainingHistoryLimit").value),
      offline_optimization: $("trainingOfflineOptimization").checked,
      optimization_trials: Number($("trainingOptimizationTrials").value),
      screening_epochs: Number($("trainingScreeningEpochs").value),
      hard_example_replay: $("trainingHardReplay").checked,
      targets: Array.from(document.querySelectorAll("[data-training-target]:checked"), (input) => input.dataset.trainingTarget),
      auto_promote: $("trainingAutoPromote").checked,
      require_quality_gate: true,
    };
  }

  async function saveTrainingSettings() {
    const button = $("saveTrainingSettings"); button.disabled = true;
    try {
      const payload = trainingSettingsPayload();
      if (!payload.targets.length) throw new Error("至少选择一个训练目标。");
      await C.api("/admin/training/settings", { method: "POST", json: payload });
      setTrainingNotice("自动训练设置已保存。只有人工审核通过的样本会进入训练；自动更新始终经过实拍质量门禁。", "");
      C.toast("自动训练设置已保存。");
      await loadTraining({ silent: true });
    } catch (error) { setTrainingNotice(`保存失败：${error.message}`, "error"); C.toast(error.message, "error"); }
    finally { button.disabled = false; }
  }

  async function startTraining() {
    if (!window.confirm("确定立即建立候选模型吗？训练会占用显卡。当前线上模型不会在质量门禁通过前被替换。")) return;
    const button = $("startTraining"); button.disabled = true;
    try {
      const payload = await C.api("/admin/training/start", { method: "POST", json: { confirm: true } });
      setTrainingNotice(`训练任务 ${payload.training_job.id} 已启动。可以离开本页面，后台会继续训练。`, "");
      C.toast("候选模型训练已启动。");
      await loadTraining({ silent: true });
    } catch (error) { setTrainingNotice(`无法启动训练：${error.message}`, "error"); C.toast(error.message, "error"); }
    finally { button.disabled = Boolean(state.training?.active_job); }
  }

  async function rollbackTrainingModel() {
    if (!window.confirm("确定回滚最近一次自动模型更新吗？页面会恢复到更新前的模型权重。")) return;
    const button = $("rollbackTrainingModel"); button.disabled = true;
    try {
      await C.api("/admin/training/rollback", { method: "POST", json: { confirm: true } });
      setTrainingNotice("最近一次自动模型更新已安全回滚，后续检测将加载上一版模型。", "");
      C.toast("模型已回滚。");
      await loadTraining({ silent: true });
    } catch (error) { setTrainingNotice(`回滚失败：${error.message}`, "error"); C.toast(error.message, "error"); }
  }

  async function loadInspections({ silent = false } = {}) {
    if (state.inspectionLoading) return;
    state.inspectionLoading = true;
    $("refreshInspections").disabled = true;
    try {
      const payload = await C.api("/admin/inspections?limit=500");
      state.inspections = payload.inspections || [];
      renderInspectionSummary(payload.summary || {});
      populateInspectionTenants(payload.tenants || []);
      renderInspectionList({ preserveDetail: silent });
      if (!silent) C.toast(`已加载 ${state.inspections.length} 条企业检测记录。`);
    } catch (error) { C.toast(error.message, "error"); }
    finally { state.inspectionLoading = false; $("refreshInspections").disabled = false; }
  }

  function renderInspectionSummary(summary) {
    $("sumAllInspections").textContent = summary.inspections || 0;
    $("sumInspectionConfirmed").textContent = summary.confirmed || 0;
    $("sumInspectionFeedback").textContent = summary.feedback_total || 0;
    $("sumInspectionFailed").textContent = summary.detection_failed || 0;
  }

  function populateInspectionTenants(tenants) {
    const select = $("inspectionTenantFilter");
    const selected = select.value;
    select.replaceChildren(new Option("全部企业", ""));
    for (const tenant of tenants) select.append(new Option(tenant.name, tenant.id));
    if ([...select.options].some((option) => option.value === selected)) select.value = selected;
  }

  function filteredInspections() {
    const tenantId = $("inspectionTenantFilter").value;
    const stateFilter = $("inspectionStateFilter").value;
    const search = $("inspectionSearch").value.trim().toLocaleLowerCase("zh-CN");
    return state.inspections.filter((item) => {
      if (tenantId && item.tenant_id !== tenantId) return false;
      if (stateFilter === "feedback" && !item.feedback_receipt) return false;
      if (stateFilter && stateFilter !== "feedback" && item.state !== stateFilter) return false;
      if (search && !String(item.filename || "").toLocaleLowerCase("zh-CN").includes(search)) return false;
      return true;
    });
  }

  function renderInspectionList({ preserveDetail = false } = {}) {
    const items = filteredInspections();
    const list = $("inspectionList");
    list.replaceChildren();
    $("inspectionListEmpty").classList.toggle("hidden", items.length > 0);
    if (state.selectedInspection) {
      state.selectedInspection = items.find((item) => item.id === state.selectedInspection.id) || null;
    }
    for (const item of items) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.inspectionId = item.id;
      button.className = `inspection-record${state.selectedInspection?.id === item.id ? " selected" : ""}`;
      const status = inspectionStateName(item);
      button.innerHTML = `<img loading="lazy" alt="" src="${escapeHtml(C.imageUrl(item.images.preview))}"><span class="inspection-record-copy"><strong>${escapeHtml(item.filename)}</strong><p>${escapeHtml(item.tenant_name || item.tenant_id)}</p><span class="inspection-record-meta"><span class="inspection-state ${escapeHtml(item.state)}">${escapeHtml(status)}</span><time>${escapeHtml(C.formatDate(item.created_at))}</time></span></span>`;
      button.addEventListener("click", () => selectInspection(item));
      list.append(button);
    }
    if (!items.length) clearInspectionSelection();
    else if (!state.selectedInspection) selectInspection(items[0]);
    else if (!preserveDetail) renderInspectionDetail(state.selectedInspection);
  }

  function clearInspectionSelection() {
    state.selectedInspection = null;
    $("inspectionResultPanel").classList.add("empty");
    $("inspectionResultEmpty").classList.remove("hidden");
    $("inspectionResultContent").classList.add("hidden");
  }

  function selectInspection(item) {
    state.selectedInspection = item;
    for (const button of $("inspectionList").children) button.classList.toggle("selected", button.dataset.inspectionId === item.id);
    renderInspectionDetail(item);
  }

  function renderInspectionDetail(item) {
    const prediction = item.prediction || {};
    const sourceSize = prediction.source_size || { width: 1, height: 1 };
    const centers = prediction.inner_line_centers_px;
    const pair = prediction.centering_pair_percent;
    $("inspectionResultPanel").classList.remove("empty");
    $("inspectionResultEmpty").classList.add("hidden");
    $("inspectionResultContent").classList.remove("hidden");
    $("inspectionResultFilename").textContent = item.filename || "—";
    $("inspectionResultTenant").textContent = item.tenant_name || item.tenant_id || "—";
    $("inspectionResultTime").textContent = C.formatDate(item.created_at);
    $("inspectionResultState").textContent = inspectionStateName(item);
    $("recordOuterOverlay").setAttribute("viewBox", `0 0 ${sourceSize.width} ${sourceSize.height}`);
    $("recordOuterOverlay").querySelector("polygon").setAttribute("points", (prediction.outer_corners || []).map((point) => point.join(",")).join(" "));
    state.recordOuterZoom?.setSourceSize(sourceSize);
    state.recordInnerZoom?.setSourceSize(prediction.rectified_size || { width: 630, height: 880 });
    $("recordOuterImage").src = C.imageUrl(item.images.preview);
    if (prediction.outer_corners) $("recordInnerImage").src = C.imageUrl(item.images.rectified);
    else $("recordInnerImage").removeAttribute("src");
    for (const line of $("recordInnerSurface").querySelectorAll("[data-inspection-edge]")) line.classList.toggle("hidden", !centers);
    if (centers) C.renderLines($("recordInnerSurface"), "data-inspection-edge", centers);
    $("inspectionHorizontal").textContent = pair ? `${formatMetric(pair.left)} / ${formatMetric(pair.right)}` : "—";
    $("inspectionVertical").textContent = pair ? `${formatMetric(pair.top)} / ${formatMetric(pair.bottom)}` : "—";
    const confidence = prediction.confidence?.overall;
    $("inspectionConfidence").textContent = confidence == null ? "—" : `${(Number(confidence) * 100).toFixed(1)}%`;
    $("inspectionDeviation").textContent = prediction.maximum_deviation_percent == null ? "—" : `${Number(prediction.maximum_deviation_percent).toFixed(2)}%`;
    $("inspectionModel").textContent = item.model_version || prediction.model_version || "—";
    $("inspectionFeedbackState").textContent = item.feedback_receipt ? `${STATUS_NAMES[item.feedback_receipt.review_status] || item.feedback_receipt.review_status} · ${item.feedback_receipt.id}` : "未反馈";
    $("openInspectionFeedback").classList.toggle("hidden", !item.feedback_receipt);
    window.requestAnimationFrame(() => { state.recordOuterZoom?.fit(); state.recordInnerZoom?.fit(); });
  }

  async function openSelectedInspectionFeedback() {
    const receipt = state.selectedInspection?.feedback_receipt;
    if (!receipt) return;
    $("statusFilter").value = "";
    state.selected = { id: receipt.id };
    await switchView("feedback");
  }

  function inspectionStateName(item) {
    return INSPECTION_STATE_NAMES[item.state] || item.state || "未知状态";
  }

  function formatMetric(value) {
    return Number(value).toFixed(1);
  }

  async function loadFeedback({ silent = false, force = false } = {}) {
    if (state.feedbackLoading) {
      if (force) state.feedbackReloadQueued = true;
      return;
    }
    if (silent && state.reviewInFlight) return;
    state.feedbackLoading = true;
    $("refreshFeedback").disabled = true;
    const previousPending = Number($("sumPending").textContent || 0);
    const filter = $("statusFilter").value;
    try {
      const payload = await C.api(`/admin/feedback${filter ? `?status=${encodeURIComponent(filter)}` : ""}`);
      state.items = payload.feedback || [];
      renderSummary(payload.summary || {});
      if (state.selected) {
        const current = state.items.find((item) => item.id === state.selected.id);
        if (current) {
          state.selected = current;
          if (silent) renderList(); else selectFeedback(current);
        } else if (state.items.length) selectFeedback(state.items[0]);
        else { clearSelection(); renderList(); }
      } else if (state.items.length) selectFeedback(state.items[0]);
      else renderList();
      const currentPending = Number(payload.summary?.feedback_pending || 0);
      if (silent && currentPending > previousPending) C.toast(`收到 ${currentPending - previousPending} 条新反馈。`);
    } catch (error) { C.toast(error.message, "error"); }
    finally {
      state.feedbackLoading = false; $("refreshFeedback").disabled = false;
      if (state.feedbackReloadQueued) {
        state.feedbackReloadQueued = false;
        window.queueMicrotask(() => loadFeedback());
      }
    }
  }

  function renderSummary(summary) {
    $("sumInspections").textContent = summary.inspections || 0;
    $("sumConfirmed").textContent = summary.confirmed || 0;
    $("sumPending").textContent = summary.feedback_pending || 0;
    $("sumApproved").textContent = summary.feedback_approved || 0;
  }

  async function loadTenants() {
    try {
      const payload = await C.api("/admin/tenants");
      state.tenants = payload.tenants || [];
      renderTenantSummary(payload.summary || {});
      renderTenants();
    } catch (error) { C.toast(error.message, "error"); }
  }

  function renderTenantSummary(summary) {
    $("sumTenants").textContent = state.tenants.length;
    $("sumActiveTenants").textContent = summary.tenants_active || 0;
    $("sumTenantInspections").textContent = summary.inspections || 0;
    $("sumTenantPending").textContent = summary.feedback_pending || 0;
  }

  function renderTenants() {
    const body = $("tenantList");
    body.replaceChildren();
    $("tenantEmpty").classList.toggle("hidden", state.tenants.length > 0);
    for (const tenant of state.tenants) {
      const row = document.createElement("tr");
      const unavailable = !tenant.active || tenant.expired;
      const status = !tenant.active ? "已停用" : tenant.expired ? "已过期" : "有效";
      row.innerHTML = `
        <td><strong>${escapeHtml(tenant.name)}</strong><small>${escapeHtml(tenant.id)}</small></td>
        <td><strong>${escapeHtml(tenant.username || "未设置")}</strong><small>${tenant.account_configured ? "密码已配置" : "需要开通账号"}</small></td>
        <td><span class="tenant-status ${unavailable ? "inactive" : "active"}">${status}</span></td>
        <td>${formatFullDate(tenant.expires_at, "长期有效")}</td>
        <td>${formatFullDate(tenant.last_used_at, "尚未使用")}</td>
        <td><strong>${Number(tenant.inspection_count || 0)}</strong><small>${Number(tenant.pending_feedback_count || 0)} 条待审核</small></td>
        <td><div class="tenant-actions">
          <button class="table-action" data-action="copy_login">复制登录地址</button>
          <button class="table-action" data-action="set_credentials">${tenant.account_configured ? "重置账号" : "开通账号"}</button>
          <button class="table-action" data-action="extend">延期一年</button>
          <button class="table-action" data-action="${tenant.active ? "disable" : "enable"}">${tenant.active ? "停用" : "启用"}</button>
          <button class="table-action danger" data-action="delete">删除企业</button>
        </div></td>`;
      for (const button of row.querySelectorAll("[data-action]")) {
        button.addEventListener("click", () => handleTenantAction(tenant, button.dataset.action, button));
      }
      body.append(row);
    }
  }

  async function createTenant(event) {
    event.preventDefault();
    const name = $("tenantName").value.trim();
    if (!name) { C.toast("请填写企业名称。", "error"); return; }
    $("submitTenant").disabled = true;
    try {
      const username = $("tenantUsername").value.trim();
      const password = $("tenantPassword").value;
      const payload = await C.api("/admin/tenants", {
        method: "POST",
        json: {
          name,
          username: username || undefined,
          password: password || undefined,
          valid_days: Number($("tenantValidDays").value),
        },
      });
      closeTenantDialog();
      showCredentials(payload.credentials, payload.tenant.name);
      C.toast(`已创建“${payload.tenant.name}”的企业账号。`);
      await loadTenants();
    } catch (error) { C.toast(error.message, "error"); }
    finally { $("submitTenant").disabled = false; }
  }

  function closeTenantDialog() {
    $("tenantDialog").close();
    $("tenantForm").reset();
  }

  function showCredentials(credentials, displayName, kind = "enterprise") {
    const adminAccount = kind === "admin";
    const text = [
      `CardScope ${adminAccount ? "标注管理员" : "企业"}登录信息`,
      `${adminAccount ? "姓名" : "企业"}：${displayName}`,
      `登录地址：${credentials.login_url}`,
      `账号：${credentials.username}`,
      `初始密码：${credentials.initial_password}`,
      "请妥善保管，不要转发给无关人员。",
    ].join("\n");
    $("credentialDialogTitle").textContent = adminAccount ? "标注管理员登录信息" : "企业登录信息";
    $("credentialDialogSubtitle").textContent = "初始密码只在本次显示，请立即安全发送给使用人。";
    $("credentialWarning").textContent = adminAccount
      ? "标注管理员只能处理人工标注任务；忘记密码时请在管理员列表中重置。"
      : "平台无法查看旧密码；如企业忘记密码，请在企业列表中重置账号。";
    $("credentialText").value = text;
    $("credentialDialog").showModal();
  }

  function closeCredentialDialog() {
    $("credentialDialog").close();
    $("credentialText").value = "";
  }

  async function handleTenantAction(tenant, action, button) {
    if (action === "copy_login") {
      try {
        await copyText(tenant.login_url);
        C.toast("企业登录地址已复制，请同时把账号密码安全发给企业。");
      } catch (error) { C.toast(error.message, "error"); }
      return;
    }
    if (action === "delete") {
      const confirmation = window.prompt(
        `此操作会永久删除“${tenant.name}”及其 ${Number(tenant.inspection_count || 0)} 条检测记录、问题反馈、图片和标注。\n\n已发布的模型不会自动回滚。请输入完整企业名称确认：`,
        ""
      );
      if (confirmation === null) return;
      if (confirmation !== tenant.name) {
        C.toast("企业名称输入不一致，已取消删除。", "error");
        return;
      }
      button.disabled = true;
      try {
        const payload = await C.api(`/admin/tenants/${tenant.id}`, {
          method: "DELETE",
          json: { confirm_name: confirmation },
        });
        const message = payload.cleanup_warning
          ? `企业已删除，但有文件残留需要技术人员清理：${payload.cleanup_warning}`
          : `“${tenant.name}”及其 ${Number(payload.deleted_counts?.inspections || 0)} 条检测数据已永久删除。`;
        C.toast(message, payload.cleanup_warning ? "error" : "success");
        await loadTenants();
      } catch (error) { button.disabled = false; C.toast(error.message, "error"); }
      return;
    }
    if (action === "set_credentials") {
      const username = window.prompt(
        "请输入企业登录账号（3–64 位字母、数字、点、下划线或短横线）：",
        tenant.username || ""
      );
      if (username === null) return;
      if (tenant.account_configured && !window.confirm(
        `确定重置“${tenant.name}”的登录密码吗？该企业当前所有已登录设备会立即退出。`
      )) return;
      button.disabled = true;
      try {
        const payload = await C.api(`/admin/tenants/${tenant.id}/action`, {
          method: "POST",
          json: { action: "set_credentials", username: username.trim() || undefined },
        });
        showCredentials(payload.credentials, tenant.name);
        await loadTenants();
      } catch (error) { button.disabled = false; C.toast(error.message, "error"); }
      return;
    }
    if (action === "rotate" && !window.confirm(`确定重新生成“${tenant.name}”的链接吗？旧链接会立即失效。`)) return;
    if (action === "disable" && !window.confirm(`确定停用“${tenant.name}”吗？其链接将暂时无法访问。`)) return;
    button.disabled = true;
    try {
      const json = { action };
      if (action === "extend") json.days = 365;
      const payload = await C.api(`/admin/tenants/${tenant.id}/action`, { method: "POST", json });
      if (action === "rotate") {
        await copyText(payload.tenant.share_url);
        C.toast("新链接已生成并复制，旧链接已失效。");
      } else {
        C.toast(action === "extend" ? "有效期已延长一年。" : action === "enable" ? "企业链接已启用。" : "企业链接已停用。");
      }
      await loadTenants();
    } catch (error) { button.disabled = false; C.toast(error.message, "error"); }
  }

  async function loadAdminUsers() {
    try {
      const payload = await C.api("/admin/users");
      state.admins = payload.admins || [];
      $("sumAdminUsers").textContent = state.admins.length;
      $("sumActiveAdminUsers").textContent = state.admins.filter((item) => item.active).length;
      renderAdminUsers(payload.login_url || "/admin-login");
    } catch (error) { C.toast(error.message, "error"); }
  }

  function renderAdminUsers(loginUrl) {
    const body = $("adminUserList");
    body.replaceChildren();
    $("adminUserEmpty").classList.toggle("hidden", state.admins.length > 0);
    for (const admin of state.admins) {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><strong>${escapeHtml(admin.label)}</strong><small>${escapeHtml(admin.id)}</small></td>
        <td><strong>${escapeHtml(admin.username)}</strong><small>独立密码登录</small></td>
        <td><span class="permission-badge">人工标注</span></td>
        <td><span class="tenant-status ${admin.active ? "active" : "inactive"}">${admin.active ? "已启用" : "已停用"}</span></td>
        <td>${formatFullDate(admin.last_used_at, "尚未使用")}</td>
        <td>${formatFullDate(admin.created_at)}</td>
        <td><div class="tenant-actions">
          <button class="table-action" data-action="copy_login">复制登录地址</button>
          <button class="table-action" data-action="reset_credentials">重置账号密码</button>
          <button class="table-action" data-action="${admin.active ? "disable" : "enable"}">${admin.active ? "停用" : "启用"}</button>
          <button class="table-action danger" data-action="delete">删除账号</button>
        </div></td>`;
      for (const button of row.querySelectorAll("[data-action]")) {
        button.addEventListener("click", () => handleAdminUserAction(admin, button.dataset.action, button, loginUrl));
      }
      body.append(row);
    }
  }

  async function createAdminUser(event) {
    event.preventDefault();
    const label = $("adminUserLabel").value.trim();
    if (!label) { C.toast("请填写管理员姓名。", "error"); return; }
    $("submitAdminUser").disabled = true;
    try {
      const username = $("adminUsername").value.trim();
      const password = $("adminPassword").value;
      const payload = await C.api("/admin/users", {
        method: "POST",
        json: { label, username: username || undefined, password: password || undefined },
      });
      closeAdminUserDialog();
      showCredentials(payload.credentials, payload.admin.label, "admin");
      C.toast(`已创建标注管理员“${payload.admin.label}”。`);
      await loadAdminUsers();
    } catch (error) { C.toast(error.message, "error"); }
    finally { $("submitAdminUser").disabled = false; }
  }

  function closeAdminUserDialog() {
    $("adminUserDialog").close();
    $("adminUserForm").reset();
  }

  async function handleAdminUserAction(admin, action, button, loginUrl) {
    if (action === "copy_login") {
      try {
        await copyText(loginUrl);
        C.toast("管理员登录地址已复制，请同时安全发送账号和密码。")
      } catch (error) { C.toast(error.message, "error"); }
      return;
    }
    if (action === "delete") {
      const confirmation = window.prompt(
        `确定永久删除标注管理员“${admin.label}”吗？该账号会立即无法登录。\n\n请输入完整登录账号确认：`, ""
      );
      if (confirmation === null) return;
      if (confirmation !== admin.username) { C.toast("登录账号输入不一致，已取消删除。", "error"); return; }
      button.disabled = true;
      try {
        await C.api(`/admin/users/${admin.id}`, { method: "DELETE", json: { confirm_username: confirmation } });
        C.toast(`标注管理员“${admin.label}”已删除。`);
        await loadAdminUsers();
      } catch (error) { button.disabled = false; C.toast(error.message, "error"); }
      return;
    }
    if (action === "reset_credentials") {
      const username = window.prompt("请输入新的管理员登录账号：", admin.username || "");
      if (username === null) return;
      if (!window.confirm(`确定重置“${admin.label}”的账号密码吗？该账号当前登录状态会立即失效。`)) return;
      button.disabled = true;
      try {
        const payload = await C.api(`/admin/users/${admin.id}/action`, {
          method: "POST", json: { action, username: username.trim() || undefined },
        });
        showCredentials(payload.credentials, payload.admin.label, "admin");
        await loadAdminUsers();
      } catch (error) { button.disabled = false; C.toast(error.message, "error"); }
      return;
    }
    if (action === "disable" && !window.confirm(`确定停用“${admin.label}”吗？该账号会立即退出登录。`)) return;
    button.disabled = true;
    try {
      await C.api(`/admin/users/${admin.id}/action`, { method: "POST", json: { action } });
      C.toast(action === "enable" ? "标注管理员账号已启用。" : "标注管理员账号已停用。")
      await loadAdminUsers();
    } catch (error) { button.disabled = false; C.toast(error.message, "error"); }
  }

  async function copyText(value) {
    if (!value) throw new Error("当前没有可用的登录地址。");
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const area = document.createElement("textarea");
    area.value = value; area.setAttribute("readonly", ""); area.style.position = "fixed"; area.style.opacity = "0";
    document.body.append(area); area.select();
    const copied = document.execCommand("copy"); area.remove();
    if (!copied) throw new Error("浏览器未允许复制，请手动复制链接。");
  }

  function formatFullDate(value, emptyLabel = "—") {
    if (!value) return emptyLabel;
    try {
      return new Intl.DateTimeFormat("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date(value));
    } catch (_) { return value; }
  }

  function renderList() {
    const list = $("feedbackList"); list.replaceChildren();
    if (!state.items.length) { list.innerHTML = '<div class="empty-list"><strong>当前没有已提交的模型反馈</strong><span>居中度超标不会自动进入；企业须完成反馈表单并收到反馈编号。</span></div>'; return; }
    for (const item of state.items) {
      const button = document.createElement("button");
      button.className = `feedback-item${state.selected?.id === item.id ? " selected" : ""}`;
      button.innerHTML = `<strong>${escapeHtml(item.filename)}</strong><p>${escapeHtml(item.tenant_name)} · ${(item.issue_tags || []).map(tagName).join("、")}</p><div class="feedback-meta"><span>${C.formatDate(item.created_at)}</span><span>${STATUS_NAMES[item.review_status] || item.review_status}</span></div>`;
      button.addEventListener("click", () => selectFeedback(item));
      list.append(button);
    }
  }

  function selectFeedback(item) {
    state.selected = item; renderList();
    setActionStatus();
    if (state.rectifiedObjectUrl) { URL.revokeObjectURL(state.rectifiedObjectUrl); state.rectifiedObjectUrl = null; }
    $("reviewPanel").classList.remove("empty"); $("reviewEmpty").classList.add("hidden"); $("reviewContent").classList.remove("hidden");
    $("reviewFilename").textContent = item.filename; $("reviewTenant").textContent = item.tenant_name; $("reviewTime").textContent = C.formatDate(item.created_at);
    $("reviewStatus").textContent = STATUS_NAMES[item.review_status] || item.review_status;
    $("reviewNotes").textContent = item.notes || "企业未填写补充说明。";
    $("adminReviewNotes").value = item.review_notes || "";
    const tags = $("reviewTags"); tags.replaceChildren();
    for (const tag of item.issue_tags || []) { const node = document.createElement("span"); node.textContent = tagName(tag); tags.append(node); }
    const hasOuterIssue = (item.issue_tags || []).includes("outer_frame_wrong");
    $("outerWarning").classList.toggle("hidden", !hasOuterIssue);
    $("adminOuterImage").src = C.imageUrl(item.images.normalized || item.images.preview);
    $("adminInnerImage").src = C.imageUrl(item.images.rectified);
    const sourceSize = item.prediction?.source_size || { width: 1, height: 1 };
    state.sourceSize = sourceSize;
    state.outerZoom?.setSourceSize(sourceSize);
    state.innerZoom?.setSourceSize({ width: 630, height: 880 });
    const outerDefaults = item.corrected_outer || item.prediction?.outer_corners || [[0, 0], [sourceSize.width - 1, 0], [sourceSize.width - 1, sourceSize.height - 1], [0, sourceSize.height - 1]];
    state.rectifiedOuterKey = outerKey(item.prediction?.outer_corners || outerDefaults);
    state.getOuterCorrection = C.outerCorrectionControls(
      $("adminOuterControls"), outerDefaults, sourceSize, $("adminOuterOverlay"), (points) => { state.correctedOuter = points; }
    );
    const innerDefaults = item.corrected_inner || item.prediction?.inner_line_centers_px || { left: 25, right: 604, top: 25, bottom: 854 };
    state.getInnerCorrection = C.correctionControls($("adminCorrectionControls"), innerDefaults, (values) => {
      state.correctedInner = values; C.renderLines($("adminInnerStage"), "data-admin-edge", values);
    }, $("adminInnerStage"));
    const locked = item.review_status !== "pending";
    setReviewControlsDisabled(locked);
    $("reopenFeedback").classList.toggle("hidden", !isOwner() || item.review_status !== "approved");
    $("reopenFeedback").disabled = item.review_status !== "approved";
    $("deleteFeedback").disabled = !isOwner();
    $("discardFeedback").classList.toggle("hidden", isOwner() || item.review_status !== "pending");
    $("discardFeedback").disabled = isOwner() || locked;
    $("toggleAnnotationMode").disabled = false;
    $("resetAnnotations").disabled = locked;
    window.requestAnimationFrame(() => { state.outerZoom?.fit(); state.innerZoom?.fit(); });
  }

  function clearSelection() {
    if (state.annotationMode) toggleAnnotationMode(false);
    state.selected = null; $("reviewPanel").classList.add("empty"); $("reviewEmpty").classList.remove("hidden"); $("reviewContent").classList.add("hidden");
  }

  async function deleteSelectedFeedback() {
    if (!state.selected || state.reviewInFlight) return;
    const item = state.selected;
    const trainingNotice = item.review_status === "approved"
      ? "该样本已进入训练池，删除时会同时移除训练池文件；但不会自动回滚已经发布的模型。\n\n"
      : "";
    if (!window.confirm(
      `${trainingNotice}确定永久删除“${item.filename}”吗？\n\n对应的反馈、检测记录、原图和标注都会删除，此操作无法撤销。`
    )) return;
    state.reviewInFlight = true;
    setReviewControlsDisabled(true);
    $("deleteFeedback").disabled = true;
    $("deleteFeedback").textContent = "正在永久删除…";
    setActionStatus("正在删除数据库记录、图片和训练池数据，请稍候…", "processing");
    try {
      const result = await C.api(`/admin/feedback/${item.id}`, {
        method: "DELETE",
        json: { confirm_feedback_id: item.id },
      });
      const message = result.cleanup_warning
        ? `样本记录已删除，但有文件残留需要技术人员清理：${result.cleanup_warning}`
        : `“${item.filename}”的反馈、检测记录、图片和标注已永久删除。`;
      clearSelection();
      showAdminNotice(message, result.cleanup_warning ? "error" : "");
      C.toast(message, result.cleanup_warning ? "error" : "success");
      await loadFeedback({ force: true });
    } catch (error) {
      setActionStatus(`删除失败：${error.message}`, "error");
      C.toast(error.message, "error");
      setReviewControlsDisabled(state.selected?.review_status !== "pending");
      $("deleteFeedback").disabled = false;
    } finally {
      state.reviewInFlight = false;
      $("deleteFeedback").textContent = "永久删除样本";
    }
  }

  async function review(action) {
    if (!state.selected || state.reviewInFlight) return;
    if (action === "reopen" && !window.confirm("确定退回待审核吗？旧标注会立即失去训练资格，图片仍会保留作审计。如果此前下载过训练 ZIP，请弃用旧 ZIP 并重新导出。")) return;
    if (action === "discard" && !window.confirm("确定舍弃这条样本吗？\n\n样本会移出标注员待审核队列且不会进入训练池；原图、检测记录和舍弃原因仍会保留，供平台主管理员审计。")) return;
    const button = action === "approve" ? $("approveFeedback") : action === "needs_annotation" ? $("needsAnnotation") : action === "reopen" ? $("reopenFeedback") : action === "discard" ? $("discardFeedback") : $("rejectFeedback");
    const feedbackId = state.selected.id;
    let payload = { action, review_notes: $("adminReviewNotes").value };
    if (action === "discard" && !payload.review_notes.trim()) payload.review_notes = "标注管理员判断该样本不适合用于训练，已舍弃。";
    if (action === "approve") {
      const correctedInner = state.getInnerCorrection?.() || state.correctedInner;
      const correctedOuter = state.getOuterCorrection?.() || state.correctedOuter;
      if (!correctedInner || !(correctedInner.left < correctedInner.right && correctedInner.top < correctedInner.bottom)) {
        setActionStatus("内框坐标无效：必须满足左 < 右、上 < 下。", "error"); return;
      }
      const outerProblem = outerDraftProblem(correctedOuter);
      if (outerProblem) { setActionStatus(`外框坐标无效：${outerProblem}`, "error"); return; }
      if (outerKey(correctedOuter) !== state.rectifiedOuterKey) {
        setActionStatus("外框已经调整。请先点击“应用外框并刷新内框图”，再确认红色内框线后批准。", "error"); return;
      }
      payload = { ...payload, corrected_inner: correctedInner, corrected_outer: correctedOuter };
    }
    state.reviewInFlight = true;
    setReviewControlsDisabled(true);
    button.textContent = action === "approve" ? "正在写入训练池…" : action === "needs_annotation" ? "正在转交…" : action === "reopen" ? "正在撤销批准…" : action === "discard" ? "正在舍弃…" : "正在驳回…";
    setActionStatus(action === "approve" ? "正在保存人工标注并验证训练池文件，请稍候…" : action === "reopen" ? "正在撤销旧训练资格并退回待审核，请稍候…" : action === "discard" ? "正在舍弃样本并保留审计记录，请稍候…" : "正在保存审核状态，请稍候…", "processing");
    try {
      const result = await C.api(`/admin/feedback/${feedbackId}/review`, {
        method: "POST",
        json: payload,
      });
      const newStatus = result.feedback?.review_status;
      const expectedStatus = action === "approve" ? "approved" : action === "needs_annotation" ? "needs_annotation" : action === "reopen" ? "pending" : action === "discard" ? "discarded" : "rejected";
      if (newStatus !== expectedStatus) throw new Error("服务器未确认新的审核状态，请刷新后重试。");
      if (action === "approve" && (newStatus !== "approved" || !result.training_feedback?.sample_id)) {
        throw new Error("服务器未返回训练池写入凭据，请刷新后重试。");
      }
      if (action === "reopen" && result.training_feedback_revocation?.training_eligible !== false) {
        throw new Error("服务器没有确认旧训练数据已经失效，请刷新后重试。");
      }
      let message = action === "approve"
        ? `反馈 ${feedbackId} 已审核通过，并成功写入训练池（${result.training_feedback.sample_id}）。`
        : action === "needs_annotation" ? `反馈 ${feedbackId} 已转入高级标注队列。` : action === "reopen" ? `反馈 ${feedbackId} 已撤销旧训练资格并退回待审核；此前下载的训练 ZIP 请弃用并重新导出。` : action === "discard" ? `样本 ${feedbackId} 已舍弃并移出待标注队列，未进入训练池。` : `反馈 ${feedbackId} 已驳回。`;
      if (result.automatic_training_job?.id) message += ` 自动训练任务 ${result.automatic_training_job.id} 已在后台启动。`;
      showAdminNotice(message);
      C.toast(message);
      state.selected = action === "reopen" ? { id: feedbackId } : null;
      if (action === "reopen") $("statusFilter").value = "pending";
      if (state.annotationMode) toggleAnnotationMode(false);
      await loadFeedback({ force: true });
    } catch (error) {
      const code = error.code && error.code !== "REQUEST_FAILED" ? `（${error.code}）` : "";
      setActionStatus(`${error.message}${code}。审核状态尚未改变，可以修正后重试。`, "error");
      showAdminNotice(`处理失败：${error.message}${code}`, "error");
      C.toast(error.message, "error");
      setReviewControlsDisabled(state.selected?.review_status !== "pending");
      if (state.selected?.review_status === "approved") $("reopenFeedback").disabled = false;
    } finally {
      state.reviewInFlight = false;
      $("approveFeedback").textContent = isOwner() ? "审核通过并加入训练池" : "提交人工标注并加入训练池";
      $("needsAnnotation").textContent = "转高级标注";
      $("discardFeedback").textContent = "舍弃本样本";
      $("rejectFeedback").textContent = "驳回";
      $("reopenFeedback").textContent = "退回待审核";
    }
  }

  async function refreshRectification() {
    if (!state.selected) return;
    $("refreshRectification").disabled = true;
    try {
      const response = await C.api(`/admin/feedback/${state.selected.id}/rectify-preview`, {
        method: "POST", raw: true,
        json: { corrected_outer: state.getOuterCorrection?.() || state.correctedOuter },
      });
      const blob = await response.blob();
      if (state.rectifiedObjectUrl) URL.revokeObjectURL(state.rectifiedObjectUrl);
      state.rectifiedObjectUrl = URL.createObjectURL(blob);
      $("adminInnerImage").src = state.rectifiedObjectUrl;
      state.rectifiedOuterKey = outerKey(state.getOuterCorrection?.() || state.correctedOuter);
      state.innerZoom?.fit();
      setActionStatus("外框校正图已刷新。现在可以直接拖动红线，并用方向键进行 0.1 px 微调。", "");
      C.toast("已按人工外框重新生成校正图，请继续确认红色内框线。");
    } catch (error) { C.toast(error.message, "error"); }
    finally { $("refreshRectification").disabled = state.reviewInFlight || state.selected?.review_status !== "pending"; }
  }

  async function exportTraining() {
    $("exportTraining").disabled = true;
    try {
      const historyLimit = $("includeHistory").checked ? Number($("historyLimit").value) : 0;
      const response = await C.api(`/admin/training-export?history_limit=${historyLimit}`, { raw: true });
      await downloadResponse(response, "CardScope_training_data.zip");
      C.toast(`训练数据已批量导出${historyLimit ? `，包含内框、外框历史样本各 ${historyLimit} 张` : ""}。`);
    } catch (error) { C.toast(error.message, "error"); }
    finally { $("exportTraining").disabled = false; }
  }

  async function exportBundle() {
    $("exportBundle").disabled = true;
    try {
      const response = await C.api("/admin/feedback-export", { raw: true });
      await downloadResponse(response, "CardScope_feedback.zip");
      C.toast("审核归档已导出。训练请使用旁边的“批量导出训练数据”。")
    } catch (error) { C.toast(error.message, "error"); }
    finally { $("exportBundle").disabled = false; }
  }

  async function downloadResponse(response, fallbackName) {
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const url = URL.createObjectURL(blob); const link = document.createElement("a");
    link.href = url; link.download = match?.[1] || fallbackName; document.body.append(link); link.click(); link.remove(); URL.revokeObjectURL(url);
  }

  function tagName(value) { return TAG_NAMES[value] || value; }
  function escapeHtml(value) { const node = document.createElement("span"); node.textContent = String(value); return node.innerHTML; }

  init();
})();
