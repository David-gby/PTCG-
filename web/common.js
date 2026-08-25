(function () {
  "use strict";
  // Keep the visible inner-frame guide sub-pixel thin at every zoom level.
  // Selection remains easy because correctionControls uses a separate 22 px hit radius.
  const INNER_LINE_SCREEN_PX = 0.8;
  const INNER_LINE_HALO_SCREEN_PX = 0.35;

  const TOKEN_KEY = "cardscope_access_token";
  const params = new URLSearchParams(window.location.search);
  const linkToken = params.get("access");
  if (linkToken) {
    sessionStorage.setItem(TOKEN_KEY, linkToken);
    history.replaceState(null, "", window.location.pathname);
  }

  function token() {
    return sessionStorage.getItem(TOKEN_KEY) || "";
  }

  function clearToken() {
    sessionStorage.removeItem(TOKEN_KEY);
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (token()) headers.set("X-Platform-Token", token());
    if (options.json !== undefined) {
      headers.set("Content-Type", "application/json");
      options.body = JSON.stringify(options.json);
    }
    const response = await fetch(`/api/platform/v1${path}`, {
      credentials: "same-origin", ...options, headers,
    });
    if (options.raw) {
      if (!response.ok) await throwResponse(response);
      return response;
    }
    let payload = null;
    try { payload = await response.json(); } catch (_) { payload = null; }
    if (!response.ok) {
      const error = new Error(payload?.error?.message || `请求失败（${response.status}）`);
      error.code = payload?.error?.code || "REQUEST_FAILED";
      error.status = response.status;
      error.details = payload?.error?.details || {};
      throw error;
    }
    return payload;
  }

  async function throwResponse(response) {
    let payload = null;
    try { payload = await response.json(); } catch (_) { payload = null; }
    const error = new Error(payload?.error?.message || `请求失败（${response.status}）`);
    error.code = payload?.error?.code || "REQUEST_FAILED";
    error.status = response.status;
    throw error;
  }

  function imageUrl(path) {
    if (!token()) return path;
    const separator = path.includes("?") ? "&" : "?";
    return `${path}${separator}access=${encodeURIComponent(token())}`;
  }

  let toastTimer = 0;
  function toast(message, type = "success") {
    const node = document.getElementById("toast");
    if (!node) return;
    window.clearTimeout(toastTimer);
    node.textContent = message;
    node.className = `toast show${type === "error" ? " error" : ""}`;
    toastTimer = window.setTimeout(() => { node.className = "toast"; }, 3200);
  }

  function formatDate(value) {
    if (!value) return "—";
    try {
      return new Intl.DateTimeFormat("zh-CN", {
        month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit",
      }).format(new Date(value));
    } catch (_) { return value; }
  }

  const EDGE_LABELS = { left: "左", right: "右", top: "上", bottom: "下" };
  const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
  function renderLines(stage, selectorPrefix, centers) {
    if (!stage || !centers) return;
    for (const edge of ["left", "right", "top", "bottom"]) {
      const line = stage.querySelector(`[${selectorPrefix}="${edge}"]`);
      if (!line) continue;
      if (edge === "left" || edge === "right") line.style.left = `${Number(centers[edge]) / 629 * 100}%`;
      else line.style.top = `${Number(centers[edge]) / 879 * 100}%`;
    }
  }

  function correctionControls(container, initial, onChange, stage = null) {
    const values = { ...initial };
    let selectedEdge = null;
    let pointerId = null;
    container.replaceChildren();
    const controls = {};
    const setSelected = (edge) => {
      selectedEdge = edge;
      if (!stage) return;
      stage.querySelectorAll("[data-admin-edge]").forEach((line) => {
        line.classList.toggle("selected", line.dataset.adminEdge === edge);
      });
    };
    const setValue = (edge, rawValue) => {
      const maximum = edge === "left" || edge === "right" ? 629 : 879;
      let numeric = Number(rawValue);
      if (!Number.isFinite(numeric)) return;
      if (edge === "left") numeric = clamp(numeric, 0, Number(values.right) - 1);
      else if (edge === "right") numeric = clamp(numeric, Number(values.left) + 1, maximum);
      else if (edge === "top") numeric = clamp(numeric, 0, Number(values.bottom) - 1);
      else numeric = clamp(numeric, Number(values.top) + 1, maximum);
      values[edge] = Math.round(numeric * 10) / 10;
      const pair = controls[edge];
      if (pair) {
        pair.range.value = String(values[edge]);
        pair.number.value = values[edge].toFixed(1);
      }
      setSelected(edge);
      onChange({ ...values });
    };
    for (const edge of ["left", "right", "top", "bottom"]) {
      const maximum = edge === "left" || edge === "right" ? 629 : 879;
      const row = document.createElement("div");
      row.className = "correction-row";
      const label = document.createElement("label");
      label.textContent = EDGE_LABELS[edge];
      const range = document.createElement("input");
      range.type = "range"; range.min = "0"; range.max = String(maximum); range.step = "0.1";
      range.value = String(values[edge]); range.dataset.edge = edge;
      const number = document.createElement("input");
      number.type = "number"; number.min = "0"; number.max = String(maximum); number.step = "0.1";
      number.value = Number(values[edge]).toFixed(1); number.dataset.edge = edge;
      range.addEventListener("input", () => setValue(edge, range.value));
      number.addEventListener("change", () => setValue(edge, number.value));
      range.addEventListener("focus", () => setSelected(edge));
      number.addEventListener("focus", () => setSelected(edge));
      row.append(label, range, number);
      container.append(row); controls[edge] = { range, number };
    }
    if (stage) {
      stage._cardscopeInnerCleanup?.();
      const surface = stage.querySelector(".annotation-surface") || stage;
      const closestEdge = (event) => {
        const bounds = surface.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return null;
        const x = (event.clientX - bounds.left) / bounds.width * 630;
        const y = (event.clientY - bounds.top) / bounds.height * 880;
        const distances = {
          left: Math.abs(x - values.left) / 630 * bounds.width,
          right: Math.abs(x - values.right) / 630 * bounds.width,
          top: Math.abs(y - values.top) / 880 * bounds.height,
          bottom: Math.abs(y - values.bottom) / 880 * bounds.height,
        };
        return Object.entries(distances).sort((a, b) => a[1] - b[1])[0];
      };
      const onPointerDown = (event) => {
        if (stage.dataset.locked === "true" || event.button !== 0 || event.shiftKey) return;
        const nearest = closestEdge(event);
        if (!nearest || nearest[1] > 22) return;
        event.preventDefault();
        pointerId = event.pointerId; setSelected(nearest[0]); stage.focus();
        stage.setPointerCapture?.(event.pointerId);
      };
      const onPointerMove = (event) => {
        if (pointerId !== event.pointerId || !selectedEdge) return;
        const bounds = surface.getBoundingClientRect();
        const value = selectedEdge === "left" || selectedEdge === "right"
          ? (event.clientX - bounds.left) / bounds.width * 630
          : (event.clientY - bounds.top) / bounds.height * 880;
        setValue(selectedEdge, value);
      };
      const onPointerEnd = (event) => {
        if (pointerId !== event.pointerId) return;
        pointerId = null;
        try { stage.releasePointerCapture?.(event.pointerId); } catch (_) { /* already released */ }
      };
      const onKeyDown = (event) => {
        if (stage.dataset.locked === "true" || !selectedEdge) return;
        const horizontal = selectedEdge === "left" || selectedEdge === "right";
        const allowed = horizontal ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
        if (!allowed.includes(event.key)) return;
        event.preventDefault();
        const direction = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
        setValue(selectedEdge, values[selectedEdge] + direction * (event.shiftKey ? 1 : 0.1));
      };
      stage.addEventListener("pointerdown", onPointerDown);
      stage.addEventListener("pointermove", onPointerMove);
      stage.addEventListener("pointerup", onPointerEnd);
      stage.addEventListener("pointercancel", onPointerEnd);
      stage.addEventListener("keydown", onKeyDown);
      stage._cardscopeInnerCleanup = () => {
        stage.removeEventListener("pointerdown", onPointerDown);
        stage.removeEventListener("pointermove", onPointerMove);
        stage.removeEventListener("pointerup", onPointerEnd);
        stage.removeEventListener("pointercancel", onPointerEnd);
        stage.removeEventListener("keydown", onKeyDown);
      };
    }
    onChange({ ...values });
    return () => ({ ...values });
  }

  const FIXED_INNER_WIDTH = 580;
  const FIXED_INNER_HEIGHT = 830;
  const RECTIFIED_WIDTH = 630;
  const RECTIFIED_HEIGHT = 880;

  function normalizeFixedInnerBox(initial) {
    const fallback = { left: 25, right: 605, top: 25, bottom: 855 };
    const source = initial && typeof initial === "object" ? initial : fallback;
    const left = Number(source.left);
    const right = Number(source.right);
    const top = Number(source.top);
    const bottom = Number(source.bottom);
    const centerX = Number.isFinite(left) && Number.isFinite(right) ? (left + right) / 2 : 315;
    const centerY = Number.isFinite(top) && Number.isFinite(bottom) ? (top + bottom) / 2 : 440;
    const originX = Math.round(clamp(centerX - FIXED_INNER_WIDTH / 2, 0, RECTIFIED_WIDTH - 1 - FIXED_INNER_WIDTH) * 10) / 10;
    const originY = Math.round(clamp(centerY - FIXED_INNER_HEIGHT / 2, 0, RECTIFIED_HEIGHT - 1 - FIXED_INNER_HEIGHT) * 10) / 10;
    return {
      left: originX,
      right: Math.round((originX + FIXED_INNER_WIDTH) * 10) / 10,
      top: originY,
      bottom: Math.round((originY + FIXED_INNER_HEIGHT) * 10) / 10,
    };
  }

  function fixedInnerBoxControls(container, initial, onChange, stage) {
    let values = normalizeFixedInnerBox(initial);
    let drag = null;
    const box = stage?.querySelector("[data-admin-inner-box]") || null;
    container.replaceChildren();

    const summary = document.createElement("div");
    summary.className = "fixed-inner-summary";
    summary.innerHTML = '<div><strong>580 × 830 px</strong><span>尺寸已锁定，只移动整个框</span></div><span class="fixed-inner-badge">印刷线内沿</span>';

    const coordinates = document.createElement("div");
    coordinates.className = "fixed-inner-coordinate-grid";
    const editableInputs = {};
    const outputs = {};
    const addCoordinate = (labelText, key, editable) => {
      const label = document.createElement("label");
      label.className = editable ? "fixed-inner-coordinate editable-coordinate" : "fixed-inner-coordinate inferred-coordinate";
      const caption = document.createElement("span");
      caption.textContent = labelText;
      const field = editable ? document.createElement("input") : document.createElement("output");
      if (editable) {
        field.type = "number"; field.min = "0"; field.max = "49"; field.step = "0.1";
        field.setAttribute("aria-label", `${labelText}坐标`);
        editableInputs[key] = field;
      } else {
        outputs[key] = field;
      }
      label.append(caption, field); coordinates.append(label);
    };
    addCoordinate("左边 X", "left", true);
    addCoordinate("上边 Y", "top", true);
    addCoordinate("右边 X（自动）", "right", false);
    addCoordinate("下边 Y（自动）", "bottom", false);
    container.append(summary, coordinates);

    const render = () => {
      if (box) {
        box.setAttribute("x", String(values.left));
        box.setAttribute("y", String(values.top));
        box.setAttribute("width", String(FIXED_INNER_WIDTH));
        box.setAttribute("height", String(FIXED_INNER_HEIGHT));
      }
      editableInputs.left.value = values.left.toFixed(1);
      editableInputs.top.value = values.top.toFixed(1);
      outputs.right.textContent = values.right.toFixed(1);
      outputs.bottom.textContent = values.bottom.toFixed(1);
      onChange({ ...values });
    };
    const setOrigin = (rawLeft, rawTop) => {
      const leftValue = Number(rawLeft);
      const topValue = Number(rawTop);
      const next = normalizeFixedInnerBox({
        left: Number.isFinite(leftValue) ? leftValue : values.left,
        right: (Number.isFinite(leftValue) ? leftValue : values.left) + FIXED_INNER_WIDTH,
        top: Number.isFinite(topValue) ? topValue : values.top,
        bottom: (Number.isFinite(topValue) ? topValue : values.top) + FIXED_INNER_HEIGHT,
      });
      values = next; render();
    };
    const updateLeftInput = () => {
      if (editableInputs.left.value !== "") setOrigin(editableInputs.left.value, values.top);
    };
    const updateTopInput = () => {
      if (editableInputs.top.value !== "") setOrigin(values.left, editableInputs.top.value);
    };
    editableInputs.left.addEventListener("input", updateLeftInput);
    editableInputs.left.addEventListener("change", updateLeftInput);
    editableInputs.top.addEventListener("input", updateTopInput);
    editableInputs.top.addEventListener("change", updateTopInput);

    if (stage && box) {
      stage._cardscopeInnerCleanup?.();
      const surface = stage.querySelector(".annotation-surface") || stage;
      const onPointerDown = (event) => {
        if (stage.dataset.locked === "true" || event.button !== 0 || event.shiftKey || event.target !== box) return;
        event.preventDefault();
        const bounds = surface.getBoundingClientRect();
        if (!bounds.width || !bounds.height) return;
        drag = {
          id: event.pointerId,
          clientX: event.clientX,
          clientY: event.clientY,
          left: values.left,
          top: values.top,
          scaleX: RECTIFIED_WIDTH / bounds.width,
          scaleY: RECTIFIED_HEIGHT / bounds.height,
        };
        box.classList.add("selected");
        stage.focus(); stage.setPointerCapture?.(event.pointerId);
      };
      const onPointerMove = (event) => {
        if (!drag || drag.id !== event.pointerId) return;
        setOrigin(
          drag.left + (event.clientX - drag.clientX) * drag.scaleX,
          drag.top + (event.clientY - drag.clientY) * drag.scaleY,
        );
      };
      const onPointerEnd = (event) => {
        if (!drag || drag.id !== event.pointerId) return;
        drag = null; box.classList.remove("selected");
        try { stage.releasePointerCapture?.(event.pointerId); } catch (_) { /* already released */ }
      };
      const onKeyDown = (event) => {
        if (stage.dataset.locked === "true" || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) return;
        event.preventDefault();
        const step = event.shiftKey ? 1 : 0.1;
        const x = values.left + (event.key === "ArrowLeft" ? -step : event.key === "ArrowRight" ? step : 0);
        const y = values.top + (event.key === "ArrowUp" ? -step : event.key === "ArrowDown" ? step : 0);
        setOrigin(x, y);
      };
      stage.addEventListener("pointerdown", onPointerDown);
      stage.addEventListener("pointermove", onPointerMove);
      stage.addEventListener("pointerup", onPointerEnd);
      stage.addEventListener("pointercancel", onPointerEnd);
      stage.addEventListener("keydown", onKeyDown);
      stage._cardscopeInnerCleanup = () => {
        stage.removeEventListener("pointerdown", onPointerDown);
        stage.removeEventListener("pointermove", onPointerMove);
        stage.removeEventListener("pointerup", onPointerEnd);
        stage.removeEventListener("pointercancel", onPointerEnd);
        stage.removeEventListener("keydown", onKeyDown);
      };
    }
    render();
    return () => ({ ...values });
  }

  function zoomPanControls(viewport, surface, initialSize, buttons = {}) {
    let sourceWidth = Math.max(1, Number(initialSize?.width || 1));
    let sourceHeight = Math.max(1, Number(initialSize?.height || 1));
    const lineScreenPx = Math.max(0.25, Number(buttons.lineScreenPx || INNER_LINE_SCREEN_PX));
    const lineHaloScreenPx = Math.max(0, Number(buttons.lineHaloScreenPx ?? INNER_LINE_HALO_SCREEN_PX));
    const view = { scale: 1, fitScale: 1, panX: 0, panY: 0, fitted: true };
    let pan = null;
    const apply = () => {
      surface.style.width = `${sourceWidth}px`;
      surface.style.height = `${sourceHeight}px`;
      surface.style.transform = `translate(${view.panX}px, ${view.panY}px) scale(${view.scale})`;
      surface.style.setProperty("--annotation-line-size", `${lineScreenPx / Math.max(view.scale, 0.0001)}px`);
      surface.style.setProperty("--annotation-halo-size", `${lineHaloScreenPx / Math.max(view.scale, 0.0001)}px`);
      viewport.dataset.annotationScale = String(view.scale);
      surface.querySelectorAll("[data-outer-corner]").forEach((circle) => {
        circle.setAttribute("r", String(7 / Math.max(view.scale, 0.0001)));
      });
      if (buttons.zoomText) {
        const relative = view.fitScale > 0 ? view.scale / view.fitScale : 1;
        buttons.zoomText.textContent = `${Math.round(relative * 100)}%`;
      }
    };
    const fit = () => {
      const width = Math.max(1, viewport.clientWidth);
      const height = Math.max(1, viewport.clientHeight);
      const padding = 16;
      view.fitScale = Math.max(0.0001, Math.min((width - padding * 2) / sourceWidth, (height - padding * 2) / sourceHeight));
      view.scale = view.fitScale;
      view.panX = (width - sourceWidth * view.scale) / 2;
      view.panY = (height - sourceHeight * view.scale) / 2;
      view.fitted = true; apply();
    };
    const zoomAt = (factor, clientX, clientY) => {
      const bounds = viewport.getBoundingClientRect();
      const x = clientX - bounds.left;
      const y = clientY - bounds.top;
      const sourceX = (x - view.panX) / view.scale;
      const sourceY = (y - view.panY) / view.scale;
      view.scale = clamp(view.scale * factor, view.fitScale * 0.4, view.fitScale * 24);
      view.panX = x - sourceX * view.scale;
      view.panY = y - sourceY * view.scale;
      view.fitted = false; apply();
    };
    const zoomCenter = (factor) => {
      const bounds = viewport.getBoundingClientRect();
      zoomAt(factor, bounds.left + bounds.width / 2, bounds.top + bounds.height / 2);
    };
    const onWheel = (event) => {
      event.preventDefault();
      zoomAt(Math.exp(-event.deltaY * 0.0015), event.clientX, event.clientY);
    };
    const onPointerDown = (event) => {
      const primaryPan = buttons.panWithPrimary === true && event.button === 0;
      if (!(primaryPan || event.button === 1 || event.shiftKey)) return;
      event.preventDefault(); event.stopImmediatePropagation();
      pan = { id: event.pointerId, x: event.clientX, y: event.clientY, panX: view.panX, panY: view.panY };
      viewport.classList.add("is-panning"); viewport.setPointerCapture?.(event.pointerId);
    };
    const onPointerMove = (event) => {
      if (!pan || pan.id !== event.pointerId) return;
      view.panX = pan.panX + event.clientX - pan.x;
      view.panY = pan.panY + event.clientY - pan.y;
      view.fitted = false; apply();
    };
    const onPointerEnd = (event) => {
      if (!pan || pan.id !== event.pointerId) return;
      pan = null; viewport.classList.remove("is-panning");
      try { viewport.releasePointerCapture?.(event.pointerId); } catch (_) { /* already released */ }
    };
    viewport.addEventListener("wheel", onWheel, { passive: false });
    viewport.addEventListener("pointerdown", onPointerDown, true);
    viewport.addEventListener("pointermove", onPointerMove, true);
    viewport.addEventListener("pointerup", onPointerEnd, true);
    viewport.addEventListener("pointercancel", onPointerEnd, true);
    buttons.fit?.addEventListener("click", fit);
    buttons.zoomIn?.addEventListener("click", () => zoomCenter(1.25));
    buttons.zoomOut?.addEventListener("click", () => zoomCenter(0.8));
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(() => {
      if (view.fitted) fit(); else apply();
    }) : null;
    observer?.observe(viewport);
    return {
      fit,
      zoomIn: () => zoomCenter(1.25),
      zoomOut: () => zoomCenter(0.8),
      setSourceSize(size) {
        sourceWidth = Math.max(1, Number(size?.width || 1));
        sourceHeight = Math.max(1, Number(size?.height || 1));
        fit();
      },
      refresh() { if (view.fitted) fit(); else apply(); },
      destroy() { observer?.disconnect(); },
    };
  }

  const CORNER_LABELS = ["左上", "右上", "右下", "左下"];
  function renderOuter(svg, points) {
    if (!svg || !Array.isArray(points) || points.length !== 4) return;
    const polygon = svg.querySelector("polygon");
    if (polygon) polygon.setAttribute("points", points.map((point) => point.join(",")).join(" "));
    svg.querySelectorAll("[data-outer-corner]").forEach((circle) => {
      const point = points[Number(circle.dataset.outerCorner)];
      if (!point) return;
      circle.setAttribute("cx", point[0]); circle.setAttribute("cy", point[1]);
    });
  }

  function outerCorrectionControls(container, initial, sourceSize, svg, onChange) {
    const width = Math.max(1, Number(sourceSize?.width || 1));
    const height = Math.max(1, Number(sourceSize?.height || 1));
    const points = initial.map((point) => [Number(point[0]), Number(point[1])]);
    const inputs = [];
    container.replaceChildren();
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const radius = 7 / Math.max(Number(svg.closest(".annotation-viewport")?.dataset.annotationScale || 1), 0.0001);
    svg.querySelectorAll("[data-outer-corner]").forEach((circle) => circle.setAttribute("r", radius));
    const emit = () => {
      renderOuter(svg, points);
      inputs.forEach(({ xInput, yInput }, index) => {
        xInput.value = points[index][0].toFixed(1); yInput.value = points[index][1].toFixed(1);
      });
      onChange(points.map((point) => [...point]));
    };
    points.forEach((point, index) => {
      const row = document.createElement("div"); row.className = "outer-correction-row";
      const label = document.createElement("label"); label.textContent = CORNER_LABELS[index];
      const xInput = document.createElement("input"); xInput.type = "number"; xInput.min = "0"; xInput.max = String(width - 1); xInput.step = "0.1";
      const yInput = document.createElement("input"); yInput.type = "number"; yInput.min = "0"; yInput.max = String(height - 1); yInput.step = "0.1";
      const update = () => {
        const xValue = Math.max(0, Math.min(width - 1, Number(xInput.value)));
        const yValue = Math.max(0, Math.min(height - 1, Number(yInput.value)));
        if (!Number.isFinite(xValue) || !Number.isFinite(yValue)) return;
        points[index] = [Math.round(xValue * 10) / 10, Math.round(yValue * 10) / 10]; emit();
      };
      xInput.addEventListener("change", update); yInput.addEventListener("change", update);
      row.append(label, document.createTextNode("X"), xInput, document.createTextNode("Y"), yInput);
      container.append(row); inputs.push({ xInput, yInput });
    });
    let active = -1; let selected = -1; let pointerId = null;
    const stage = svg.closest(".annotation-viewport");
    const selectCorner = (index) => {
      selected = index;
      svg.querySelectorAll("[data-outer-corner]").forEach((circle) => {
        circle.classList.toggle("selected", Number(circle.dataset.outerCorner) === index);
      });
    };
    svg.querySelectorAll("[data-outer-corner]").forEach((circle) => {
      circle.onpointerdown = (event) => {
        if (stage?.dataset.locked === "true" || event.button !== 0 || event.shiftKey) return;
        active = Number(circle.dataset.outerCorner); pointerId = event.pointerId;
        selectCorner(active); stage?.focus();
        svg.setPointerCapture?.(event.pointerId); event.preventDefault();
      };
    });
    svg.onpointermove = (event) => {
      if (active < 0) return;
      const bounds = svg.getBoundingClientRect();
      const xValue = Math.max(0, Math.min(width - 1, (event.clientX - bounds.left) / bounds.width * width));
      const yValue = Math.max(0, Math.min(height - 1, (event.clientY - bounds.top) / bounds.height * height));
      points[active] = [Math.round(xValue * 10) / 10, Math.round(yValue * 10) / 10]; emit();
    };
    const release = (event) => {
      if (pointerId !== null && event?.pointerId !== undefined && pointerId !== event.pointerId) return;
      active = -1; pointerId = null;
    };
    svg.onpointerup = release; svg.onpointercancel = release;
    if (stage) {
      stage._cardscopeOuterKeyCleanup?.();
      const onKeyDown = (event) => {
        if (stage.dataset.locked === "true" || selected < 0) return;
        const directions = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
        if (!directions[event.key]) return;
        event.preventDefault();
        const step = event.shiftKey ? 1 : 0.2;
        points[selected][0] = Math.round(clamp(points[selected][0] + directions[event.key][0] * step, 0, width - 1) * 10) / 10;
        points[selected][1] = Math.round(clamp(points[selected][1] + directions[event.key][1] * step, 0, height - 1) * 10) / 10;
        emit();
      };
      stage.addEventListener("keydown", onKeyDown);
      stage._cardscopeOuterKeyCleanup = () => stage.removeEventListener("keydown", onKeyDown);
    }
    emit();
    return () => points.map((point) => [...point]);
  }

  window.CardScope = { api, token, clearToken, imageUrl, toast, formatDate, renderLines, correctionControls, normalizeFixedInnerBox, fixedInnerBoxControls, renderOuter, outerCorrectionControls, zoomPanControls };
})();
