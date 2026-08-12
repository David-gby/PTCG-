(function () {
  "use strict";
  const C = window.CardScope;
  const form = document.getElementById("loginForm");
  const username = document.getElementById("loginUsername");
  const password = document.getElementById("loginPassword");
  const button = document.getElementById("loginButton");
  const errorNode = document.getElementById("loginError");
  const toggle = document.getElementById("togglePassword");

  function showError(message) {
    errorNode.textContent = message;
    errorNode.classList.remove("hidden");
  }

  async function detectExistingSession() {
    try {
      const payload = await C.api("/session");
      if (payload?.session?.role === "enterprise") window.location.replace("/enterprise");
    } catch (_) {
      // The expected state for a visitor who has not logged in yet.
    }
  }

  toggle.addEventListener("click", () => {
    const reveal = password.type === "password";
    password.type = reveal ? "text" : "password";
    toggle.textContent = reveal ? "隐藏" : "显示";
    toggle.setAttribute("aria-label", reveal ? "隐藏密码" : "显示密码");
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    errorNode.classList.add("hidden");
    button.disabled = true;
    button.textContent = "正在验证…";
    try {
      C.clearToken();
      const payload = await C.api("/auth/login", {
        method: "POST",
        json: { username: username.value.trim(), password: password.value },
      });
      if (payload?.session?.role !== "enterprise") throw new Error("该账号不是企业账号。");
      window.location.replace("/enterprise");
    } catch (error) {
      showError(error.message || "登录失败，请检查账号和密码。");
      password.select();
    } finally {
      button.disabled = false;
      button.textContent = "登录并开始检测";
    }
  });

  detectExistingSession();
})();
