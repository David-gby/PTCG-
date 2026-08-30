(function () {
  "use strict";
  const C = window.CardScope;
  const state = {
    session: null, current: null,
    batch: [], batchIndex: -1, history: [], feedbackReceipts: new Map(), feedbackDrafts: new Map(),
    outerZoom: null, innerZoom: null,
    activeBatch: null, batchPollTimer: 0, batchUploading: false, batchResultsLoaded: false,
    prefetchedImages: new Map(),
    historyTotal: 0, historyHasMore: false, historyLoading: false, resultCollectionSource: "",
  };
  const $ = (id) => document.getElementById(id);

  const nodes = {
    upload: $("uploadPanel"), file: $("fileInput"), choose: $("chooseButton"), processing: $("processingPanel"),
    referenceUpload: $("referenceUploadPanel"), referenceCapture: $("referenceCaptureInput"),
    referenceStandard: $("referenceStandardInput"), referenceStart: $("referenceUploadButton"),
    processingTitle: $("processingTitle"), processingDetail: $("processingDetail"), result: $("resultPanel"),
    rectifiedStage: $("rectifiedStage"), rectifiedImage: $("rectifiedImage"), originalStage: $("originalStage"),
    originalImage: $("originalImage"), outerOverlay: $("outerOverlay"),
    feedbackDetails: $("feedbackDetails"), feedbackDetailsCount: $("feedbackDetailsCount"),
    feedbackDetailsHelp: $("feedbackDetailsHelp"),
    queue: $("batchQueuePanel"), resumeFile: $("resumeFileInput"),
  };

  async function init() {
    bindEvents();
    setupInspectionTools();
    try {
      const payload = await C.api("/session");
      state.session = payload.session;
      if (state.session.role !== "enterprise") throw new Error("此链接不是企业检测链接。");
      $("accountName").textContent = state.session.display_name;
      $("accountAvatar").textContent = state.session.display_name.slice(0, 1);
      await loadHistory();
      await restoreBatchJob();
    } catch (error) {
      if (error.status === 401) {
        C.clearToken();
        window.location.replace("/login");
        return;
      }
      C.toast(error.message, "error");
      nodes.upload.innerHTML = `<h2>访问链接无效</h2><p>${escapeHtml(error.message)}</p>`;
    }
  }

  function setupInspectionTools() {
    state.outerZoom = C.zoomPanControls($("outerInspectionViewport"), nodes.originalStage, { width: 1, height: 1 }, {
      fit: $("outerInspectionFit"), zoomOut: $("outerInspectionZoomOut"), zoomIn: $("outerInspectionZoomIn"),
      zoomText: $("outerInspectionZoomText"), panWithPrimary: true,
    });
    state.innerZoom = C.zoomPanControls($("innerInspectionViewport"), nodes.rectifiedStage, { width: 630, height: 880 }, {
      fit: $("innerInspectionFit"), zoomOut: $("innerInspectionZoomOut"), zoomIn: $("innerInspectionZoomIn"),
      zoomText: $("innerInspectionZoomText"), panWithPrimary: true, lineScreenPx: 0.7, lineHaloScreenPx: 0.15,
    });
    nodes.originalImage.addEventListener("load", () => state.outerZoom?.fit());
    nodes.rectifiedImage.addEventListener("load", () => state.innerZoom?.fit());
  }

  function bindEvents() {
    $("traditionalModeButton").addEventListener("click", () => selectDetectionMode("traditional"));
    $("referenceModeButton").addEventListener("click", () => selectDetectionMode("reference"));
    nodes.referenceStart.addEventListener("click", uploadReferencePair);
    nodes.choose.addEventListener("click", () => nodes.file.click());
    nodes.file.addEventListener("change", () => uploadFiles([...nodes.file.files]));
    nodes.resumeFile.addEventListener("change", () => resumeBatchUpload([...nodes.resumeFile.files]));
    for (const event of ["dragenter", "dragover"]) nodes.upload.addEventListener(event, (e) => { e.preventDefault(); nodes.upload.classList.add("dragging"); });
    for (const event of ["dragleave", "drop"]) nodes.upload.addEventListener(event, (e) => { e.preventDefault(); nodes.upload.classList.remove("dragging"); });
    nodes.upload.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));
    $("newInspection").addEventListener("click", resetInspect);
    $("confirmButton").addEventListener("click", confirmCurrent);
    $("feedbackInner").addEventListener("click", () => quickSubmitFeedback(["inner_frame_wrong"], $("feedbackInner")));
    $("feedbackOuter").addEventListener("click", () => quickSubmitFeedback(["outer_frame_wrong"], $("feedbackOuter")));
    $("feedbackBoth").addEventListener("click", () => quickSubmitFeedback(["inner_frame_wrong", "outer_frame_wrong"], $("feedbackBoth")));
    nodes.feedbackDetails.addEventListener("input", () => {
      const value = nodes.feedbackDetails.value;
      nodes.feedbackDetailsCount.textContent = `${value.length} / 2000`;
      if (state.current) state.feedbackDrafts.set(state.current.id, value);
    });
    $("previousResult").addEventListener("click", () => moveResult(-1));
    $("nextResult").addEventListener("click", () => moveResult(1));
    document.querySelectorAll(".nav-item[data-view]").forEach((button) => button.addEventListener("click", () => selectView(button.dataset.view)));
    $("refreshHistory").addEventListener("click", () => loadHistory());
    $("historyFilter").addEventListener("change", () => loadHistory());
    $("loadMoreHistory").addEventListener("click", () => loadHistory({ append: true }));
    $("resumeUploadButton").addEventListener("click", () => nodes.resumeFile.click());
    $("pauseBatchButton").addEventListener("click", () => batchAction("pause"));
    $("resumeBatchButton").addEventListener("click", () => batchAction("resume"));
    $("retryBatchButton").addEventListener("click", () => batchAction("retry"));
    $("cancelBatchButton").addEventListener("click", cancelBatch);
    $("viewBatchResultsButton").addEventListener("click", loadBatchResults);
    $("logoutButton").addEventListener("click", logout);
    document.addEventListener("keydown", handleResultShortcut);
  }

  function selectDetectionMode(mode) {
    const reference = mode === "reference";
    nodes.upload.classList.toggle("hidden", reference);
    nodes.referenceUpload.classList.toggle("hidden", !reference);
    $("traditionalModeButton").className = `button ${reference ? "quiet" : "primary"}`;
    $("referenceModeButton").className = `button ${reference ? "primary" : "quiet"}`;
  }

  async function uploadReferencePair() {
    const capture = nodes.referenceCapture.files[0];
    const reference = nodes.referenceStandard.files[0];
    if (!capture || !reference) {
      C.toast("请同时选择实拍卡牌和标准图。", "error");
      return;
    }
    if (!supportedImage(capture) || !supportedImage(reference)) {
      C.toast("仅支持 JPG、PNG、WebP 图片。", "error");
      return;
    }
    nodes.referenceStart.disabled = true;
    nodes.processing.classList.remove("hidden");
    nodes.processingTitle.textContent = "正在上传实拍图和标准图";
    nodes.processingDetail.textContent = "两张图片上传完成后将开始参考图配准。";
    try {
      const created = await C.api("/reference-inspections", {
        method: "POST",
        json: { capture: { filename: capture.name }, reference: { filename: reference.name } },
      });
      const jobId = created.job.id;
      await C.api(`/reference-inspections/${jobId}/capture`, {
        method: "POST", headers: { "Content-Type": inferredContentType(capture) }, body: capture,
      });
      const completed = await C.api(`/reference-inspections/${jobId}/reference`, {
        method: "POST", headers: { "Content-Type": inferredContentType(reference) }, body: reference,
      });
      if (completed.inspection) {
        setResultCollection([completed.inspection], 0, "reference");
        C.toast("参考图配准完成。" );
      }
    } catch (error) {
      C.toast(error.message, "error");
    } finally {
      nodes.processing.classList.add("hidden");
      nodes.referenceStart.disabled = false;
      nodes.referenceCapture.value = "";
      nodes.referenceStandard.value = "";
    }
  }

  async function logout() {
    const button = $("logoutButton");
    button.disabled = true;
    try {
      await C.api("/auth/logout", { method: "POST", json: {} });
    } catch (_) {
      // Clearing the local legacy token still makes the browser leave the workspace.
    }
    C.clearToken();
    window.location.replace("/login");
  }

  const MAX_BATCH_FILES = 500;
  const MAX_BATCH_FILE_BYTES = 100 * 1024 * 1024;
  const MAX_BATCH_TOTAL_BYTES = 5 * 1024 * 1024 * 1024;
  const TERMINAL_BATCH_STATES = new Set(["completed", "partial", "failed", "cancelled"]);

  function activeBatchStorageKey() {
    return `cardscope_active_batch_${state.session?.tenant_id || "enterprise"}`;
  }

  function supportedImage(file) {
    const type = String(file.type || "").toLowerCase();
    if (["image/jpeg", "image/png", "image/webp"].includes(type)) return true;
    return /\.(jpe?g|png|webp)$/i.test(file.name || "");
  }

  function inferredContentType(file) {
    if (file.type) return file.type.toLowerCase();
    if (/\.png$/i.test(file.name)) return "image/png";
    if (/\.webp$/i.test(file.name)) return "image/webp";
    return "image/jpeg";
  }

  async function digestText(value) {
    const bytes = new TextEncoder().encode(value);
    if (window.crypto?.subtle) {
      const digest = await window.crypto.subtle.digest("SHA-256", bytes);
      return [...new Uint8Array(digest)].map((item) => item.toString(16).padStart(2, "0")).join("");
    }
    let hash = 2166136261;
    for (const byte of bytes) hash = Math.imul(hash ^ byte, 16777619) >>> 0;
    return `fallback_${hash.toString(16).padStart(8, "0")}_${bytes.length}`;
  }

  async function prepareFileEntries(files) {
    const occurrences = new Map();
    const entries = [];
    for (const file of files) {
      const relative = file.webkitRelativePath || file.name;
      const identity = `${relative}\u0000${file.size}\u0000${file.lastModified || 0}`;
      const occurrence = occurrences.get(identity) || 0;
      occurrences.set(identity, occurrence + 1);
      entries.push({
        file,
        clientKey: await digestText(`${identity}\u0000${occurrence}`),
      });
    }
    return entries;
  }

  function validateBatchFiles(files) {
    const accepted = files.filter(supportedImage);
    if (!accepted.length) throw new Error("请选择 JPG、PNG 或 WebP 图片。");
    if (accepted.length > MAX_BATCH_FILES) throw new Error(`一次最多上传 ${MAX_BATCH_FILES} 张图片。`);
    let total = 0;
    for (const file of accepted) {
      if (!file.size) throw new Error(`${file.name} 是空文件。`);
      if (file.size > MAX_BATCH_FILE_BYTES) throw new Error(`${file.name} 超过单张 100 MB 限制。`);
      total += file.size;
    }
    if (total > MAX_BATCH_TOTAL_BYTES) throw new Error("本批图片超过 5 GB，请拆分后上传。");
    if (accepted.length !== files.length) C.toast(`已忽略 ${files.length - accepted.length} 个非图片文件。`, "error");
    return accepted;
  }

  async function uploadFiles(files) {
    try {
      if (state.activeBatch && !TERMINAL_BATCH_STATES.has(state.activeBatch.state)) {
        renderBatchJob(state.activeBatch);
        throw new Error("当前已有批量任务，请先完成、继续上传或取消该任务。");
      }
      const accepted = validateBatchFiles(files);
      const entries = await prepareFileEntries(accepted);
      nodes.upload.classList.add("hidden");
      nodes.result.classList.add("hidden");
      nodes.processing.classList.add("hidden");
      nodes.queue.classList.remove("hidden");
      $("queueTitle").textContent = `正在建立 ${entries.length} 张图片的任务`;
      $("queueDetail").textContent = "清单建立后会逐张安全上传，检测在服务器后台继续。";
      const payload = await C.api("/batches", {
        method: "POST",
        json: {
          items: entries.map(({ file, clientKey }) => ({
            client_key: clientKey,
            filename: file.name,
            content_type: inferredContentType(file),
            size: file.size,
          })),
        },
      });
      state.activeBatch = payload.batch;
      state.batchResultsLoaded = false;
      localStorage.setItem(activeBatchStorageKey(), state.activeBatch.id);
      renderBatchJob(state.activeBatch);
      await uploadBatchEntries(entries);
    } catch (error) {
      C.toast(error.message, "error");
      if (!state.activeBatch) {
        nodes.queue.classList.add("hidden");
        nodes.upload.classList.remove("hidden");
      }
    } finally {
      nodes.file.value = "";
    }
  }

  async function uploadBatchEntries(entries) {
    if (!state.activeBatch || state.batchUploading) return;
    const batchId = state.activeBatch.id;
    const byKey = new Map(entries.map((entry) => [entry.clientKey, entry.file]));
    const pending = state.activeBatch.items
      .filter((item) => item.state === "waiting_upload" && byKey.has(item.client_key))
      .map((item) => ({ item, file: byKey.get(item.client_key) }));
    const unmatched = Number(state.activeBatch.counts?.waiting_upload || 0) - pending.length;
    if (!pending.length) {
      if (unmatched > 0) C.toast(`还有 ${unmatched} 张未匹配，请重新选择创建任务时的原图片。`, "error");
      scheduleBatchPoll(300);
      return;
    }
    state.batchUploading = true;
    renderBatchJob(state.activeBatch);
    let cursor = 0;
    let uploadFailures = 0;
    const worker = async () => {
      while (cursor < pending.length) {
        const index = cursor;
        cursor += 1;
        const { item, file } = pending[index];
        $("queueCurrentFile").textContent = file.name;
        try {
          const payload = await C.api(`/batches/${batchId}/items/${item.id}`, {
            method: "POST",
            headers: { "Content-Type": inferredContentType(file) },
            body: file,
          });
          if (state.activeBatch?.id === batchId) {
            state.activeBatch = payload.batch;
            renderBatchJob(state.activeBatch);
          }
        } catch (error) {
          uploadFailures += 1;
          C.toast(`${file.name}：${error.message}`, "error");
        }
      }
    };
    const uploadWorkers = Math.min(3, pending.length);
    await Promise.all(Array.from({ length: uploadWorkers }, () => worker()));
    state.batchUploading = false;
    nodes.resumeFile.value = "";
    await refreshBatchJob(batchId);
    if (uploadFailures) {
      C.toast(`${uploadFailures} 张上传失败，可点击“继续上传”重新选择原图片。`, "error");
    } else {
      C.toast("图片已安全送达服务器，后台检测将继续运行。");
    }
  }

  async function resumeBatchUpload(files) {
    try {
      if (!state.activeBatch) throw new Error("没有可继续的批量任务。");
      const accepted = validateBatchFiles(files);
      const entries = await prepareFileEntries(accepted);
      await uploadBatchEntries(entries);
    } catch (error) {
      C.toast(error.message, "error");
    } finally {
      nodes.resumeFile.value = "";
    }
  }

  async function restoreBatchJob() {
    const storedId = localStorage.getItem(activeBatchStorageKey());
    try {
      let batch = null;
      if (storedId) {
        try {
          batch = (await C.api(`/batches/${storedId}`)).batch;
        } catch (_) {
          localStorage.removeItem(activeBatchStorageKey());
        }
      }
      if (!batch) {
        const payload = await C.api("/batches?limit=10");
        batch = (payload.batches || []).find((item) => !TERMINAL_BATCH_STATES.has(item.state)) || null;
      }
      if (!batch) return;
      state.activeBatch = batch;
      localStorage.setItem(activeBatchStorageKey(), batch.id);
      renderBatchJob(batch);
      if (TERMINAL_BATCH_STATES.has(batch.state) && Number(batch.counts?.completed || 0)) {
        await loadBatchResults();
      } else {
        scheduleBatchPoll(500);
      }
    } catch (error) {
      C.toast(`批量任务恢复失败：${error.message}`, "error");
    }
  }

  function scheduleBatchPoll(delay = 1800) {
    window.clearTimeout(state.batchPollTimer);
    if (!state.activeBatch || TERMINAL_BATCH_STATES.has(state.activeBatch.state)) return;
    state.batchPollTimer = window.setTimeout(() => refreshBatchJob(state.activeBatch.id), delay);
  }

  async function refreshBatchJob(batchId, includeResults = false) {
    if (!batchId || state.activeBatch?.id !== batchId) return;
    try {
      const suffix = includeResults ? "?include=results" : "";
      const payload = await C.api(`/batches/${batchId}${suffix}`);
      if (state.activeBatch?.id !== batchId) return;
      state.activeBatch = payload.batch;
      renderBatchJob(state.activeBatch);
      if (TERMINAL_BATCH_STATES.has(state.activeBatch.state)) {
        await loadHistory();
        if (Number(state.activeBatch.counts?.completed || 0) && !state.batchResultsLoaded) {
          await loadBatchResults();
        }
      } else {
        scheduleBatchPoll();
      }
    } catch (error) {
      $("queueHelp").textContent = `暂时无法刷新进度：${error.message}。平台会继续后台检测。`;
      scheduleBatchPoll(4000);
    }
  }

  function renderBatchJob(batch) {
    const counts = batch.counts || {};
    const expected = Number(batch.expected_count || 0);
    const uploaded = Number(batch.uploaded_count || 0);
    const processed = Number(batch.processed_count || 0);
    const completed = Number(counts.completed || 0);
    const failed = Number(counts.failed || 0);
    const waiting = Number(counts.waiting_upload || 0);
    const terminal = TERMINAL_BATCH_STATES.has(batch.state);
    nodes.queue.classList.remove("hidden");
    nodes.processing.classList.add("hidden");
    nodes.upload.classList.toggle("hidden", !terminal);
    $("queueTitle").textContent = terminal
      ? `本批 ${expected} 张图片处理结束`
      : `后台正在处理 ${expected} 张图片`;
    $("queueDetail").textContent = batchStateDetail(batch.state);
    $("queueStateBadge").textContent = batchStateText(batch.state);
    $("queueStateBadge").className = `queue-state ${batch.state}`;
    $("uploadProgressText").textContent = `${uploaded} / ${expected}`;
    $("detectionProgressText").textContent = `${processed} / ${expected}`;
    $("uploadProgressBar").style.width = `${expected ? uploaded / expected * 100 : 0}%`;
    $("detectionProgressBar").style.width = `${expected ? processed / expected * 100 : 0}%`;
    $("queueCompletedCount").textContent = completed;
    $("queueFailedCount").textContent = failed;
    $("queueWaitingCount").textContent = waiting;
    const current = batch.items?.find((item) => item.state === "processing")
      || batch.items?.find((item) => item.state === "queued")
      || batch.items?.find((item) => item.state === "waiting_upload");
    if (!state.batchUploading) $("queueCurrentFile").textContent = current?.filename || "—";
    $("resumeUploadButton").classList.toggle("hidden", waiting === 0 || batch.state === "cancelled");
    $("pauseBatchButton").classList.toggle("hidden", terminal || batch.state === "paused");
    $("resumeBatchButton").classList.toggle("hidden", batch.state !== "paused");
    $("retryBatchButton").classList.toggle("hidden", failed === 0);
    $("viewBatchResultsButton").classList.toggle("hidden", completed === 0);
    $("cancelBatchButton").classList.toggle("hidden", terminal);
    $("queueHelp").textContent = waiting
      ? `还有 ${waiting} 张图片尚未送达服务器；重新选择创建任务时的原图片即可续传。`
      : terminal
        ? "已完成的检测结果永久保存在本企业记录中，可逐张复核。"
        : "图片已送达服务器；现在可以关闭或刷新网页，后台检测不会中断。";
  }

  function batchStateText(value) {
    return ({
      uploading: "上传中", queued: "排队中", processing: "检测中", paused: "已暂停",
      completed: "全部完成", partial: "部分失败", failed: "检测失败", cancelled: "已取消",
    })[value] || value;
  }

  function batchStateDetail(value) {
    return ({
      uploading: "部分图片仍在上传；已上传图片会同时进入后台检测。",
      queued: "图片已安全保存，正在等待模型处理。",
      processing: "模型正在逐张检测；关闭网页不会中断。",
      paused: "新图片暂不进入检测；当前正在处理的一张完成后会暂停。",
      completed: "全部图片检测完成，可以逐张查看内外框结果。",
      partial: "部分图片检测失败，可点击重试失败图片。",
      failed: "本批图片未获得有效结果，可重试或联系平台管理员。",
      cancelled: "剩余任务已经取消，已完成结果仍保留。",
    })[value] || "正在更新批量任务状态。";
  }

  async function batchAction(action) {
    if (!state.activeBatch) return;
    try {
      const payload = await C.api(`/batches/${state.activeBatch.id}/action`, {
        method: "POST", json: { action },
      });
      state.activeBatch = payload.batch;
      renderBatchJob(state.activeBatch);
      if (!TERMINAL_BATCH_STATES.has(state.activeBatch.state)) scheduleBatchPoll(300);
    } catch (error) {
      C.toast(error.message, "error");
    }
  }

  async function cancelBatch() {
    if (!state.activeBatch) return;
    if (!window.confirm("确定取消尚未完成的上传和检测吗？已完成结果会保留。")) return;
    await batchAction("cancel");
  }

  async function loadBatchResults() {
    if (!state.activeBatch) return;
    try {
      const payload = await C.api(`/batches/${state.activeBatch.id}?include=results`);
      state.activeBatch = payload.batch;
      const results = state.activeBatch.items.map((item) => item.inspection).filter(Boolean);
      state.batchResultsLoaded = true;
      renderBatchJob(state.activeBatch);
      if (results.length) {
        setResultCollection(results, 0, "batch");
        nodes.result.scrollIntoView({ behavior: "smooth", block: "start" });
      } else {
        C.toast("当前批次还没有可查看的检测结果。", "error");
      }
    } catch (error) {
      C.toast(`检测结果加载失败：${error.message}`, "error");
    }
  }

  function resultImageUrl(inspection, variant) {
    const displayKey = variant === "preview" ? "display_preview" : "display_rectified";
    const path = inspection?.images?.[displayKey] || inspection?.images?.[variant] || "";
    return path ? C.imageUrl(path) : "";
  }

  function setResultImage(image, url) {
    if (!url || image.dataset.cardscopeSource === url) return;
    image.decoding = "async";
    image.fetchPriority = "high";
    image.dataset.cardscopeSource = url;
    image.src = url;
  }

  function prefetchInspection(inspection) {
    const variants = inspection?.prediction?.outer_corners
      ? ["preview", "rectified"]
      : ["preview"];
    for (const variant of variants) {
      const url = resultImageUrl(inspection, variant);
      if (!url || state.prefetchedImages.has(url)) continue;
      const image = new Image();
      image.decoding = "async";
      image.fetchPriority = "low";
      image.src = url;
      state.prefetchedImages.set(url, image);
    }
    while (state.prefetchedImages.size > 12) {
      state.prefetchedImages.delete(state.prefetchedImages.keys().next().value);
    }
  }

  function scheduleNeighborPrefetch(inspectionId) {
    const start = () => {
      if (state.current?.id !== inspectionId) return;
      for (const index of [state.batchIndex + 1, state.batchIndex - 1]) {
        if (index >= 0 && index < state.batch.length) {
          prefetchInspection(state.batch[index]);
        }
      }
    };
    const images = [nodes.originalImage, nodes.rectifiedImage].filter(
      (image) => image.src && !image.complete
    );
    const whenIdle = () => {
      if (window.requestIdleCallback) {
        window.requestIdleCallback(start, { timeout: 1200 });
      } else {
        window.setTimeout(start, 250);
      }
    };
    if (!images.length) {
      whenIdle();
      return;
    }
    let remaining = images.length;
    const ready = () => {
      remaining -= 1;
      if (remaining > 0) return;
      whenIdle();
    };
    images.forEach((image) => {
      image.addEventListener("load", ready, { once: true });
      image.addEventListener("error", ready, { once: true });
    });
  }

  function renderInspection(inspection) {
    state.current = inspection;
    const prediction = inspection.prediction || {};
    const centers = prediction.inner_line_centers_px;
    const pair = prediction.centering_pair_percent;
    nodes.upload.classList.add("hidden"); nodes.processing.classList.add("hidden"); nodes.result.classList.remove("hidden");
    $("resultFilename").textContent = inspection.filename;
    const isReferenceRegistration = prediction.measurement_mode === "reference_registration";
    const referenceView = window.CardScopeReferenceResult?.getReferenceRegistrationView(prediction) || null;
    $("detectionModeLabel").textContent = isReferenceRegistration ? "参考图配准（标准图对比）" : "内外框检测";
    const jsonExport = $("exportResultJson");
    jsonExport.classList.toggle("hidden", !inspection.result_json_url);
    jsonExport.href = inspection.result_json_url || "#";
    $("modelVersion").textContent = `模型 ${inspection.model_version || "—"}`;
    const size = prediction.source_size || { width: 1, height: 1 };
    state.outerZoom?.setSourceSize(size);
    state.innerZoom?.setSourceSize({ width: 630, height: 880 });
    nodes.outerOverlay.setAttribute("viewBox", `0 0 ${size.width} ${size.height}`);
    const reasonCodes = Array.isArray(prediction.reason_codes) ? prediction.reason_codes : [];
    const outerGeometryInvalid = reasonCodes.includes("INVALID_KEYPOINT_GEOMETRY");
    const hasOuter = Boolean(prediction.outer_corners?.length === 4);
    const usableOuter = Boolean(hasOuter && !outerGeometryInvalid);
    const polygon = nodes.outerOverlay.querySelector("polygon");
    polygon.setAttribute("points", usableOuter
      ? prediction.outer_corners.map((point) => point.join(",")).join(" ")
      : "");
    setResultImage(nodes.originalImage, resultImageUrl(inspection, "preview"));
    window.requestAnimationFrame(() => { state.outerZoom?.fit(); state.innerZoom?.fit(); });
    const hasInner = Boolean(centers && pair);
    $("outerDetectionState").textContent = outerGeometryInvalid ? "候选无效" : usableOuter ? "已检测" : "检测失败";
    $("outerDetectionState").className = usableOuter ? "detected" : "failed";
    nodes.rectifiedStage.querySelectorAll("[data-edge]").forEach((line) => {
      line.hidden = Boolean(referenceView || !hasInner);
    });
    if (usableOuter) {
      setResultImage(nodes.rectifiedImage, resultImageUrl(inspection, "rectified"));
    } else {
      nodes.rectifiedImage.dataset.cardscopeSource = "";
      nodes.rectifiedImage.removeAttribute("src");
    }
    if (referenceView) {
      setMetric("leftValue", referenceView.horizontalPair.left); setMetric("rightValue", referenceView.horizontalPair.right);
      setMetric("topValue", referenceView.verticalPair.top); setMetric("bottomValue", referenceView.verticalPair.bottom);
      $("horizontalRatio").style.width = `${referenceView.horizontalPair.left}%`;
      $("verticalRatio").style.width = `${referenceView.verticalPair.top}%`;
      $("verdictText").textContent = referenceView.verdictText;
      $("verdictHint").textContent = referenceView.hint;
      $("verdictIcon").textContent = referenceView.icon;
      $("verdictIcon").className = `verdict-icon ${referenceView.iconClass}`;
      $("confidenceValue").textContent = referenceView.confidenceText;
      $("deviationValue").textContent = referenceView.deviationText;
      const confirmable = inspection.state === "completed";
      $("confirmButton").disabled = !confirmable;
      $("confirmButton").textContent = inspection.state === "confirmed" ? "本张已确认" : "配准结果正确，确认本张";
      $("innerDetectionState").textContent = referenceView.innerStatus;
      $("innerDetectionState").className = "detected";
    } else if (hasInner) {
      C.renderLines(nodes.rectifiedStage, "data-edge", centers);
      setMetric("leftValue", pair.left); setMetric("rightValue", pair.right);
      setMetric("topValue", pair.top); setMetric("bottomValue", pair.bottom);
      $("horizontalRatio").style.width = `${pair.left}%`;
      $("verticalRatio").style.width = `${pair.top}%`;
      const passed = Boolean(prediction.centering_passed);
      $("verdictText").textContent = passed ? "居中度达标" : "居中度超出标准";
      $("verdictHint").textContent = passed
        ? "请检查红、绿线是否贴合；检测线正确即可确认。"
        : "数值超出阈值不等于模型出错，请先检查红、绿线是否贴合。";
      $("verdictIcon").textContent = passed ? "✓" : "!";
      $("verdictIcon").className = `verdict-icon ${passed ? "success" : "review"}`;
      $("deviationValue").textContent = `${Number(prediction.maximum_deviation_percent).toFixed(2)}%`;
      const confidence = prediction.confidence?.overall;
      $("confidenceValue").textContent = confidence == null ? "—" : `${(confidence * 100).toFixed(1)}%`;
      const confirmable = inspection.state === "completed";
      $("confirmButton").disabled = !confirmable;
      $("confirmButton").textContent = inspection.state === "confirmed" ? "本张已确认" : inspection.state === "feedback_pending" ? "模型反馈审核中" : "检测线正确，确认本张";
      $("innerDetectionState").textContent = "已检测";
      $("innerDetectionState").className = "detected";
    } else {
      for (const id of ["leftValue", "rightValue", "topValue", "bottomValue", "confidenceValue", "deviationValue"]) $(id).textContent = "—";
      $("verdictText").textContent = "未获得完整结果"; $("verdictHint").textContent = "模型没有返回完整检测线，请提交模型问题反馈。"; $("verdictIcon").textContent = "!"; $("verdictIcon").className = "verdict-icon review";
      $("confirmButton").disabled = true;
      $("confirmButton").textContent = "结果不完整，请提交反馈";
      $("innerDetectionState").textContent = "检测失败";
      $("innerDetectionState").className = "failed";
    }
    const receipt = state.feedbackReceipts.get(inspection.id) || inspection.feedback_receipt?.id;
    const hasFeedback = Boolean(receipt);
    const submittedNotes = inspection.feedback_receipt?.notes || "";
    const feedbackDraft = state.feedbackDrafts.has(inspection.id)
      ? state.feedbackDrafts.get(inspection.id)
      : submittedNotes;
    state.feedbackDrafts.set(inspection.id, feedbackDraft);
    nodes.feedbackDetails.value = feedbackDraft;
    nodes.feedbackDetailsCount.textContent = `${feedbackDraft.length} / 2000`;
    nodes.feedbackDetails.disabled = hasFeedback;
    nodes.feedbackDetailsHelp.textContent = hasFeedback
      ? "说明已随反馈提交；如需修改，请联系平台管理员。"
      : "请描述偏移方向、位置和大致距离。";
    for (const button of [$("feedbackInner"), $("feedbackOuter"), $("feedbackBoth")]) button.disabled = hasFeedback;
    $("feedbackReceipt").classList.toggle("hidden", !hasFeedback);
    $("feedbackReceiptId").textContent = receipt ? `反馈编号：${receipt}` : "等待管理员复核";
    $("feedbackActionStatus").textContent = hasFeedback
      ? `已提交：${(inspection.feedback_receipt?.issue_tags || []).map(issueName).join("、") || "模型问题"}`
      : "";
    updateBatchControls();
    scheduleNeighborPrefetch(inspection.id);
  }

    function setMetric(id, value) { $(id).textContent = Number(value).toFixed(2); }

  function resetInspect() {
    state.current = null; state.batch = []; state.batchIndex = -1;
    state.feedbackDrafts.clear();
    nodes.result.classList.add("hidden");
    nodes.processing.classList.add("hidden");
    if (state.activeBatch && !TERMINAL_BATCH_STATES.has(state.activeBatch.state)) {
      nodes.queue.classList.remove("hidden");
      nodes.upload.classList.add("hidden");
      renderBatchJob(state.activeBatch);
      return;
    }
    state.activeBatch = null;
    state.batchResultsLoaded = false;
    window.clearTimeout(state.batchPollTimer);
    localStorage.removeItem(activeBatchStorageKey());
    nodes.queue.classList.add("hidden");
    nodes.upload.classList.remove("hidden");
  }

  async function confirmCurrent() {
    if (!state.current) return;
    $("confirmButton").disabled = true;
    try {
      const payload = await C.api(`/inspections/${state.current.id}/confirm`, { method: "POST", json: {} });
      replaceCurrent(payload.inspection); renderInspection(payload.inspection);
      C.toast("检测结果已确认，不会自动进入训练集。"); await loadHistory();
    } catch (error) { $("confirmButton").disabled = false; C.toast(error.message, "error"); }
  }

  async function quickSubmitFeedback(issues, sourceButton) {
    if (!state.current) return;
    const buttons = [$("feedbackInner"), $("feedbackOuter"), $("feedbackBoth")];
    buttons.forEach((button) => { button.disabled = true; });
    const originalLabel = sourceButton.textContent;
    const notes = nodes.feedbackDetails.value.trim();
    sourceButton.textContent = "正在提交…";
    $("feedbackActionStatus").textContent = "正在写入反馈审核台…";
    try {
      const payload = await C.api(`/inspections/${state.current.id}/feedback`, {
        method: "POST",
        json: { issue_tags: issues, notes },
      });
      state.feedbackReceipts.set(state.current.id, payload.feedback.id);
      replaceCurrent(payload.inspection);
      renderInspection(payload.inspection);
      C.toast(payload.message); await loadHistory();
    } catch (error) {
      $("feedbackActionStatus").textContent = `提交失败：${error.message}`;
      C.toast(error.message, "error");
    } finally {
      sourceButton.textContent = originalLabel;
      const submitted = Boolean(state.current && (state.feedbackReceipts.get(state.current.id) || state.current.feedback_receipt?.id));
      buttons.forEach((button) => { button.disabled = submitted; });
    }
  }

  const HISTORY_PAGE_SIZE = 50;

  async function loadHistory({ append = false } = {}) {
    if (state.historyLoading) return;
    state.historyLoading = true;
    const button = $("loadMoreHistory");
    const originalLabel = button.textContent;
    button.disabled = true;
    button.textContent = append ? "正在加载…" : originalLabel;
    try {
      const offset = append ? state.history.length : 0;
      const status = $("historyFilter").value;
      const payload = await C.api(
        `/inspections?limit=${HISTORY_PAGE_SIZE}&offset=${offset}&status=${encodeURIComponent(status)}`
      );
      const incoming = payload.inspections || [];
      const known = append ? new Set(state.history.map((item) => item.id)) : new Set();
      state.history = append
        ? [...state.history, ...incoming.filter((item) => !known.has(item.id))]
        : incoming;
      state.historyTotal = Number(payload.pagination?.total ?? state.history.length);
      state.historyHasMore = Boolean(payload.pagination?.has_more);
      renderHistory();
      if (append) C.toast(`已继续加载 ${incoming.length} 条记录。`);
    } catch (error) {
      C.toast(error.message, "error");
    } finally {
      state.historyLoading = false;
      button.disabled = !state.historyHasMore;
      button.textContent = originalLabel;
    }
  }

  function renderHistory() {
    const items = state.history;
    $("historyCount").textContent = `共 ${state.historyTotal} 条，已显示 ${items.length} 条`;
    const list = $("historyList"); list.replaceChildren();
    $("historyPager").classList.toggle("hidden", !items.length || !state.historyHasMore);
    $("historyPageHint").textContent = `已显示 ${items.length} / ${state.historyTotal} 条`;
    $("loadMoreHistory").disabled = !state.historyHasMore || state.historyLoading;
    if (!items.length) {
      list.innerHTML = '<div class="empty-list"><strong>当前筛选下没有记录</strong><span>可切换“全部记录”或点击刷新。</span></div>';
      return;
    }
    items.forEach((item, index) => {
      const row = document.createElement("button"); row.className = "history-row";
      row.innerHTML = `<strong>${escapeHtml(item.filename)}</strong><span>${formatPair(item.prediction?.centering_pair_percent)}</span><span>${C.formatDate(item.created_at)}</span><span class="history-state ${item.state}">${stateText(item.state)}</span>`;
      row.addEventListener("click", () => { selectView("inspect"); setResultCollection(state.history, index, "history"); });
      list.append(row);
    });
  }

  function setResultCollection(items, index, source = "") {
    state.batch = [...items];
    state.batchIndex = Math.max(0, Math.min(index, state.batch.length - 1));
    state.resultCollectionSource = source;
    if (state.batch.length) renderInspection(state.batch[state.batchIndex]);
  }

  function replaceCurrent(inspection) {
    const index = state.batch.findIndex((item) => item.id === inspection.id);
    if (index >= 0) state.batch[index] = inspection;
    const historyIndex = state.history.findIndex((item) => item.id === inspection.id);
    if (historyIndex >= 0) state.history[historyIndex] = inspection;
    state.current = inspection;
  }

  async function moveResult(direction) {
    if (!state.batch.length) return;
    let target = state.batchIndex + direction;
    if (
      direction > 0
      && target >= state.batch.length
      && state.resultCollectionSource === "history"
      && state.historyHasMore
    ) {
      const previousLength = state.history.length;
      await loadHistory({ append: true });
      if (state.history.length > previousLength) {
        state.batch = [...state.history];
        target = state.batchIndex + direction;
      }
    }
    if (target < 0 || target >= state.batch.length) return;
    state.batchIndex = target;
    renderInspection(state.batch[state.batchIndex]);
    nodes.result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function updateBatchControls() {
    const multiple = state.batch.length > 1;
    $("batchMeta").classList.toggle("hidden", !multiple);
    $("batchNavigator").classList.toggle("hidden", !multiple);
    if (!state.batch.length) return;
    const position = state.batchIndex + 1;
    $("batchPosition").textContent = `第 ${position} / ${state.batch.length} 张`;
    $("batchProgressText").textContent = `第 ${position} 张，共 ${state.batch.length} 张`;
    $("batchState").textContent = stateText(state.current?.state || "completed");
    $("batchState").className = `batch-state ${state.current?.state || "completed"}`;
    $("previousResult").disabled = state.batchIndex <= 0;
    $("nextResult").disabled = (
      state.batchIndex >= state.batch.length - 1
      && !(state.resultCollectionSource === "history" && state.historyHasMore)
    );
  }

  function handleResultShortcut(event) {
    if (event.altKey || event.ctrlKey || event.metaKey) return;
    if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) return;
    if (event.key === "ArrowLeft") moveResult(-1);
    if (event.key === "ArrowRight") moveResult(1);
  }

  function selectView(view) {
    $("inspectView").classList.toggle("hidden", view !== "inspect"); $("historyView").classList.toggle("hidden", view !== "history");
    document.querySelectorAll(".nav-item[data-view]").forEach((button) => button.classList.toggle("selected", button.dataset.view === view));
    $("viewTitle").textContent = view === "inspect" ? "卡牌居中度检测" : "检测记录";
    $("viewSubtitle").textContent = view === "inspect" ? "上传卡牌照片，自动识别内外框并计算居中度" : "查看本企业通过此链接提交的检测";
  }

  function stateText(value) { return ({ completed: "待确认", confirmed: "已确认", feedback_pending: "反馈待审核", feedback_approved: "已进入训练池", feedback_needs_annotation: "需高级标注", feedback_discarded: "样本未采用", feedback_rejected: "反馈已驳回", detection_failed: "检测失败" })[value] || value; }
    function formatPair(pair) { return pair ? `左右 ${Number(pair.left).toFixed(2)} / ${Number(pair.right).toFixed(2)}` : "未获得居中度"; }
  function issueName(value) { return ({ inner_frame_wrong: "内框问题", outer_frame_wrong: "外框问题" })[value] || value; }
  function escapeHtml(value) { const node = document.createElement("span"); node.textContent = String(value); return node.innerHTML; }

  init();
})();
