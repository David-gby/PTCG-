# CardScope 服务器 Git 管理指南

本文档说明如何用 git 管理服务器上的生产代码（`/home/ubuntu/cardscope`），让本地直接推送代码到服务器并上线，无需 GitHub 中转。

## 一、当前架构

```
你的本地 ──git push──→ 服务器 /home/ubuntu/cardscope（生产代码）
                              │
                       cardscope-home 服务从这里运行
```

- **生产代码位置**：`/home/ubuntu/cardscope`（git 仓库）
- **生产服务**：`cardscope-home.service`（systemd），WorkingDirectory 指向主目录
- **业务数据**：`/var/lib/cardscope/platform_workspace`（账号、提交、上传图，**不参与 git**）
- **GitHub（`ptcg` 远端）**：保留作远程备份，可选

## 二、一次性配置

以下配置只需在首次使用前执行一次，已完成的可以跳过。

### 1. 服务器允许 push 更新当前分支

在服务器 `/home/ubuntu/cardscope` 执行：

```bash
cd /home/ubuntu/cardscope
git config receive.denyCurrentBranch updateInstead
git config --get receive.denyCurrentBranch   # 应输出 updateInstead
```

### 2. 添加本地公钥（免密登录）

在本地 Git Bash 查看公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制输出的 `ssh-ed25519 AAAA...` 整行，在服务器上执行：

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "ssh-ed25519 AAAA...你的公钥..." >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

测试免密登录：

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@134.175.83.65 'echo OK'
```

直接输出 `OK` 即成功。

### 3. 本地添加 server 远端

在本地执行：

```bash
git remote add server ubuntu@134.175.83.65:/home/ubuntu/cardscope
git remote -v    # 确认有两个: ptcg(GitHub) + server
```

### 4. 首次 push 并验证

```bash
git push server main
```

push 成功后，服务器工作副本会自动更新（`updateInstead` 机制）。

## 三、日常开发流程

### 开始工作前（每次）

```bash
git pull server main        # 拉最新，避免和同伴冲突
```

### 改完代码，推送

```bash
git add <改动的文件>          # 只加你改的源码文件，不要用 git add . 一把梭
git commit -m "描述改动"
git push server main         # 直推服务器，工作副本自动更新
```

### 上线（手动重启服务）

```bash
ssh ubuntu@134.175.83.65 'cd /home/ubuntu/cardscope && sudo systemctl restart cardscope-home'
```

### 验证服务正常

```bash
curl -fsS http://127.0.0.1:8765/api/platform/v1/health
```

应返回 `{"status":"ok",...}`。

## 四、同伴加入协作

同伴需要配置自己的免密登录，才能在本地直接 push：

1. 同伴在本地生成密钥（如已有可跳过）：`ssh-keygen -t ed25519`
2. 同伴把 `~/.ssh/id_ed25519.pub` 内容发给你
3. 在服务器上把同伴公钥追加进 `authorized_keys`：

```bash
echo "ssh-ed25519 AAAA...同伴的公钥..." >> ~/.ssh/authorized_keys
```

4. 同伴本地执行：`git remote add server ubuntu@134.175.83.65:/home/ubuntu/cardscope`

之后同伴即可按「第三节」流程推送。

## 五、常见问题

### push 被拒绝（服务器有未提交改动）

`updateInstead` 会在服务器工作区不干净时拒绝推送，保护数据：

```bash
# 在服务器上查看是什么改动
ssh ubuntu@134.175.83.65 'cd /home/ubuntu/cardscope && git status'
# 若确认是数据文件误入，加入 .gitignore；若是有用改动，先提交
```

### 上线后功能没变化

确认重启了服务：

```bash
sudo systemctl restart cardscope-home
```

注意：git push 只更新文件，**不会自动重启**，需手动执行。

### 想备份到 GitHub

```bash
git push ptcg main
```

## 六、必须遵守的规矩

| 规矩 | 原因 |
|---|---|
| 别提交数据文件（`platform_workspace`、`uploads`、`ml_backend/models` 的 .pt） | 已在 gitignore，`git add .` 可能误加 |
| 服务器上别直接改代码文件 | 会产生脏 diff，下次 push 冲突 |
| 推送前先 pull | 避免和同伴的改动分叉 |
| 上线前重启服务 | push 不会自动重启 |
