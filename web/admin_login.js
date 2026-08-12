(function () {
  "use strict";
  const C = window.CardScope;
  const form = document.getElementById("adminLoginForm");
  const username = document.getElementById("adminLoginUsername");
  const password = document.getElementById("adminLoginPassword");
  const button = document.getElementById("adminLoginButton");
  const errorNode = document.getElementById("adminLoginError");
  const toggle = document.getElementById("toggleAdminPassword");

  function showError(message) {
    errorNode.textContent = message;
    errorNode.classList.remove("hidden");
  }

  async function detectExistingSession() {
    try {
      const payload = await C.api("/session");
      if (payload?.session?.role === "admin") window.location.replace("/admin");
    } catch (_) {
      // Expected when there is no active administrator session.
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
      if (payload?.session?.role !== "admin") throw new Error("该账号不是管理员账号。请使用企业登录入口。");
      window.location.replace("/admin");
    } catch (error) {
      showError(error.message || "登录失败，请检查账号和密码。");
      password.select();
    } finally {
      button.disabled = false;
      button.textContent = "登录管理台";
    }
  });

  detectExistingSession();
})();
