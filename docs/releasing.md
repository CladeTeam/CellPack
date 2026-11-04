# CI 与发行维护

仓库：<https://github.com/CladeTeam/CellPack>。当前发行流程面向 Linux；Python wheel 支持 CPython 3.10–3.15、x86_64 / aarch64，使用 manylinux_2_28 基线。Rust 核心 crate 为 `cellpack`，Python 包为 `pycellpack`；`crates/pycellpack` 绑定 crate 保持禁止上传 crates.io。

## 日常 CI

`.github/workflows/ci.yml` 在 main 推送、PR 和手动操作时执行：

- 校验 Rust / Python / Cargo.lock 版本一致。
- x86_64 与 ARM64 的 Rust 核心测试、文档测试、Python 绑定测试和 workspace Clippy。仅 x86_64 额外编译依赖 AVX-512 的基准示例；ARM64 检查 library 与 integration tests。
- Rust 1.89 的核心 crate 最低版本编译检查。
- Python 3.10 / 3.12 / 3.15 测试，以及 Python 3.12 的 PyTorch CPU 集成测试。

Clippy 使用 workspace 已声明的 deny 规则。现有 Rust 源码尚未统一通过 rustfmt，因此没有添加会阻断所有提交的全库格式门禁。GPU 路径不在 GitHub CPU runner 的验证范围内。

## 仅构建候选产物

打开 Actions → **Release candidate and publish** → **Run workflow**，选择 main，保留 `publish=false`、`draft_release=false`。也可执行：

```sh
gh workflow run release.yml --repo CladeTeam/CellPack --ref main \
  -f publish=false -f draft_release=false
```

工作流先校验版本，复用完整 CI，构建 12 个 wheel、一个 Python sdist 和一个 Rust `.crate`，测试安装 wheel 与从 sdist 重建安装，检查归档内容与 Twine 元数据，并生成 `SHA256SUMS`。产物保留 14 天，下载 `release-bundle` 即可审阅。此模式无需任何发行令牌，也不会创建标签、GitHub Release 或上传包。

## 发行版本与标签

准备发行时，在 main 同时更新根 `Cargo.toml` 的 workspace 版本、`pyproject.toml` 的 Python 版本及 `Cargo.lock` 中两个本地包的版本。正式版使用 `X.Y.Z`；预发行版 Rust 使用 `X.Y.Z-alpha.N` / `-beta.N` / `-rc.N`，Python 对应 `X.Y.ZaN` / `bN` / `rcN`。

审阅并提交后，由维护者创建与 Rust 版本严格匹配的 `vX.Y.Z` 标签。标签必须指向已包含在 origin/main 的提交。推送 `v*` 标签只会运行验证、构建候选产物和创建 **draft GitHub Release**，不会上传 PyPI / crates.io，也不会公开发布 GitHub Release。

正式上传是另外一次明确操作：在该版本标签上手动运行工作流，设置 `publish=true`。校验和、标签、完整 CI、全部构建均须通过，再由维护者审批 `release` environment。两个令牌必须同时存在，才会开始任何注册表上传。GitHub Release 草稿仍需维护者审阅并手动发布。

不要在目前的配置阶段执行上述正式上传操作。PyPI 与 crates.io 的上传不具备跨服务事务性：如果 Rust 已成功而 PyPI 上传失败，先检查两个注册表的实际状态；不要盲目重跑上传步骤。下载已校验的同一次 `release-bundle`，由维护者仅补传缺少的 Python 产物；已经发布的版本不可覆盖。

## GitHub 凭据与审批

仓库 Settings → Environments → **release**：

- 仅允许 `v*` 标签部署。
- 必须由 `krkawzq` 审批，禁止管理员绕过。维护者可审批自己手动发起的发行。
- `PYPI_API_TOKEN` 与 `CARGO_REGISTRY_TOKEN` 仅存放在该 environment 的 secrets。

`PYPI_API_TOKEN` 已从开发盒 `~/.env` 的 `PYPI_TOKEN` 通过标准输入配置。令牌内容不进入源码或命令参数。首次发布新项目可能需要账户级 PyPI 令牌；项目建立后应换为仅限 `pycellpack` 的项目令牌。配置 secret 不代表令牌已通过发布认证，当前没有执行上传验证。

### 配置 Rust token

1. 用将持有 crate 的账户登录 <https://crates.io/settings/tokens>，创建 API token。
2. 首次创建 crate 需要 `publish-new`；后续发布需要 `publish-update`。选择相应发布权限、合理有效期，并将 crate 范围限定到 `cellpack`（以页面支持的设置为准），无需授予 yank 或 owner 管理权限。
3. 在本机终端运行以下命令，并在交互提示中粘贴 token：

   ```sh
   gh secret set CARGO_REGISTRY_TOKEN --repo CladeTeam/CellPack --env release
   ```

   也可以在 GitHub 的 release environment 中手动添加同名 secret。不要将 token 发到聊天或提交到仓库。

4. 使用 `gh secret list --repo CladeTeam/CellPack --env release` 检查名称是否存在。该操作不会显示令牌内容，也不会发布。

crate 名称全站唯一；正式发行前需要确认 `cellpack` 名称及发布账户权限。GitHub organization 的仓库权限不会自动赋予 crates.io 或 PyPI 的所有权。凭据、包名与授权的最终验证留在维护者明确批准正式发行时完成。

参考：[Cargo 发行文档](https://doc.rust-lang.org/cargo/reference/publishing.html)、[GitHub environment](https://docs.github.com/en/actions/how-tos/deploy/configure-and-manage-deployments/manage-environments)。
