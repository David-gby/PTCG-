# CardScope 服务器部署交接说明

本目录用于把 CardScope v0.7.0 部署到企业提供的 Ubuntu 服务器。程序、模型与业务数据分离：

- 程序发布目录：`/opt/cardscope/releases/<版本>`，当前版本软链接为 `/opt/cardscope/current`。
- Python 环境：`/opt/cardscope/venv`。
- 持久业务数据：`/var/lib/cardscope/platform_workspace`。
- 配置：`/etc/cardscope/cardscope.env`。
- 临时缓存：`/var/cache/cardscope`。

因此以后更新网页、算法或模型时，只替换程序发布目录，不会覆盖企业账号、检测记录、反馈、标注和训练池。

## 一、服务器要求

- Ubuntu 22.04 LTS 或 24.04 LTS，x86_64。
- 最低 4 核 CPU、16GB 内存；本包默认为 CPU 推理。
- 系统盘与数据盘合计建议不少于 100GB。现有历史数据约 23GB，50GB 磁盘虽然可能装得下，但没有足够的更新、缓存和备份余量。
- 公网 IPv4、可解析到该 IP 的域名，以及开放 TCP 80/443。
- 部署者拥有 `sudo` 权限；建议用 SSH 密钥登录。

> CPU 服务器会明显改善公网访问、上传、缩略图加载、反馈同步和多人并发稳定性，但单张深度学习推理速度不保证比带 NVIDIA 显卡的本地电脑更快。若实测推理仍慢，应先看队列与 CPU 使用率，再决定增加 CPU 进程或升级 GPU。

## 二、收到的文件

1. `CardScope_Server_Deploy_*.zip`：可转交给部署同学的程序、正式模型、网页和部署脚本。
2. `CardScope_PRIVATE_State_Core_*.zip`：数据库与访问凭据，含企业账号及后台权限，只能私下发送给可信部署者。
3. `SHA256SUMS.txt`：传输完成后校验文件完整性。
4. 历史图片不重复压缩，使用 `Windows_上传历史数据到服务器.ps1` 从原电脑直接传输。

## 三、首次部署

以下命令中的域名和压缩包名称按实际情况替换：

```bash
sudo apt-get update
sudo apt-get install -y unzip
unzip CardScope_Server_Deploy_v0.7.0_20260729.zip
cd CardScope_Server_Deploy_v0.7.0_20260729
sudo bash 02_deployment_scripts/install_ubuntu.sh 01_application https://cards.example.com
```

安装程序会创建独立服务账号、Python 虚拟环境、systemd 服务和持久数据目录。应用只监听 `127.0.0.1:8765`，公网入口必须由 Caddy/Nginx 提供 HTTPS。

安装 Caddy 后，将 `02_deployment_scripts/Caddyfile.example` 中的域名替换成正式域名，再执行：

```bash
sudo cp 02_deployment_scripts/Caddyfile.example /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

## 四、迁移账号、反馈、标注和历史图片

1. 在原电脑结束当天新增任务，预留一次短维护窗口，避免迁移期间继续上传造成两边数据不一致。
2. 把私密包上传到服务器并解压到部署账号家目录的迁移暂存目录。
3. 在原电脑运行 `Windows_上传历史数据到服务器.ps1`，直接上传 `studio_data`、`upload_batches` 与自动训练状态。
4. 确认暂存目录结构为：

```text
cardscope_migration/
  platform_workspace/
    studio_data/
    private/
      platform.sqlite3
      access_links.json
      upload_batches/
      auto_training/
```

5. 在服务器执行：

```bash
sudo bash 02_deployment_scripts/finalize_migration.sh "$HOME/cardscope_migration/platform_workspace"
```

6. 验证企业登录、管理员登录、历史记录、反馈审核和新图片检测。确认服务器无误后，原电脑不再作为生产平台写入；管理员和标注员都改用服务器域名登录，以服务器数据为唯一正式数据。

## 五、日常运维

```bash
sudo systemctl status cardscope --no-pager
sudo journalctl -u cardscope -n 200 --no-pager
sudo systemctl restart cardscope
curl -fsS http://127.0.0.1:8765/api/platform/v1/health
```

管理入口：`https://正式域名/admin-login`

企业入口：`https://正式域名/login`

管理员可在任何联网电脑上使用，不要求原电脑保持开机。标注管理员也用自己的账号从浏览器登录。

## 六、以后更新网页、算法或模型

把新的服务器发布压缩包上传后执行：

```bash
sudo bash /opt/cardscope/current/deployment/server/update_release.sh /path/to/new_bundle.zip https://cards.example.com
```

脚本会先保留旧版本，再切换新版本并调用健康检查；失败时自动回滚。更新不会覆盖 `/var/lib/cardscope/platform_workspace`。正式更新前仍建议执行：

```bash
sudo bash /opt/cardscope/current/deployment/server/backup_state.sh
```

## 七、上线验收

- [ ] 域名 HTTPS 证书正常，HTTP 自动跳转 HTTPS。
- [ ] `/api/platform/v1/health` 返回成功。
- [ ] 企业账号可以登录、上传并查看批量任务进度。
- [ ] 管理员可看到企业检测、反馈、标注和训练池。
- [ ] 新反馈提交后，后台状态立即变化。
- [ ] 删除、驳回、批准、退回和舍弃样本均实测通过。
- [ ] 数据库 `PRAGMA integrity_check` 为 `ok`。
- [ ] 已配置云盘快照或异地备份。
- [ ] 原电脑端不再继续写入旧生产库。
