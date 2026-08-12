(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  if (root) root.CardScopeReferenceResult = api;
})(typeof window === "undefined" ? globalThis : window, function () {
  "use strict";

  function signedDirection(value, positive, negative, neutral) {
    if (value > 0.005) return positive;
    if (value < -0.005) return negative;
    return neutral;
  }

  function rounded(value) {
    return Number(Number(value).toFixed(4));
  }

  function getReferenceRegistrationView(prediction) {
    if (prediction?.measurement_mode !== "reference_registration") return null;
    const offset = prediction.offset || {};
    const dx = Number(offset.dx_px || 0);
    const dy = Number(offset.dy_px || 0);
    const toleranceX = Number(offset.tolerance_px?.x || 0);
    const toleranceY = Number(offset.tolerance_px?.y || toleranceX);
    const width = Math.max(2, Number(prediction.image_size?.width) || 630);
    const height = Math.max(2, Number(prediction.image_size?.height) || 880);
    const horizontalDeviation = dx / (width - 1) * 100;
    const verticalDeviation = dy / (height - 1) * 100;
    const horizontalPair = {
      left: rounded(50 + horizontalDeviation),
      right: rounded(50 - horizontalDeviation),
    };
    const verticalPair = {
      top: rounded(50 + verticalDeviation),
      bottom: rounded(50 - verticalDeviation),
    };
    const maximumDeviationPercent = rounded(Math.max(Math.abs(horizontalDeviation), Math.abs(verticalDeviation)));
    const toleranceText = toleranceX === toleranceY
      ? `容差 ±${toleranceX.toFixed(2)} px`
      : `水平容差 ±${toleranceX.toFixed(2)} px、垂直容差 ±${toleranceY.toFixed(2)} px`;
    const withinTolerance = Boolean(offset.within_tolerance);

    return {
      innerStatus: "配准完成",
      verdictText: "参考图配准结果",
      hint: `相对标准图：${signedDirection(dx, "向右", "向左", "水平")} ${Math.abs(dx).toFixed(2)} px，${signedDirection(dy, "向下", "向上", "垂直")} ${Math.abs(dy).toFixed(2)} px（${toleranceText}）。`,
      icon: withinTolerance ? "✓" : "!",
      iconClass: withinTolerance ? "success" : "review",
      confidenceText: prediction.confidence == null ? "—" : `${(Number(prediction.confidence) * 100).toFixed(1)}%`,
      deviationText: `${maximumDeviationPercent.toFixed(2)}%`,
      horizontalPair,
      verticalPair,
      maximumDeviationPercent,
    };
  }

  return { getReferenceRegistrationView };
});
