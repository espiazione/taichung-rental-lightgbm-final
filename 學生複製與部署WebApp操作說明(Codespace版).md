# 學生複製與部署臺中租金預測 WebApp 操作說明（Codespace 版）

版本日期：2026-05-24

本文件提供課堂使用的操作流程，目標是讓學生不必先處理各自電腦的 Python、GeoPandas、GDAL、Git LFS、作業系統差異等環境問題，而是統一使用 GitHub Codespaces 進行開發與部署。

本課堂建議主流程：

```text
GitHub Fork
→ GitHub Codespaces 開發
→ hf upload 部署到 HuggingFace Space
→ 修改程式或資料後，分別更新 GitHub 與 HuggingFace Space
```

範本 GitHub repository：

```text
https://github.com/chingmu-kuroro/taichung-rental-lightgbm
```

範本 HuggingFace Space：

```text
https://huggingface.co/spaces/Chingmu/taichung-rental-lightgbm
```


## 一、為什麼課堂建議使用 Codespaces？

本 WebApp 使用 Python、Solara、leafmap、GeoPandas、LightGBM、SHAP、GeoPackage 圖資與 Git LFS。若每位學生都在自己的電腦安裝環境，常見問題包括：

- Windows、macOS、Linux 指令不同。
- Python 版本不同。
- GeoPandas、pyogrio、rtree、GDAL 或空間套件安裝失敗。
- Git LFS 沒有安裝，導致 `.gpkg` 或 `.joblib` 下載不完整。
- 本機硬體或權限設定不同。
- 學生不熟悉命令列與環境變數。

使用 GitHub Codespaces 的好處是：

- 直接在瀏覽器中開啟 VS Code 類似介面。
- 每位學生的開發環境較一致。
- 不需要先 clone 到本機。
- 不需要先安裝 Python 套件到學生電腦。
- 可以直接在 Codespace 終端機執行 Git、Python、Solara、HuggingFace CLI。
- 修改後可直接 `git push` 回自己的 GitHub repository。
- 可直接 `hf upload` 部署到自己的 HuggingFace Space。


## 二、學生需要準備的帳號

每位學生需要：

1. GitHub 帳號。
2. HuggingFace 帳號。

請先確認能登入：

```text
https://github.com
https://huggingface.co
```

另外，HuggingFace 部署需要 Access Token。建議學生到 HuggingFace 建立一個具有 write 權限的 token：

```text
https://huggingface.co/settings/tokens
```

注意：token 不可以寫進程式碼、不可以 commit 到 GitHub，也不要貼在公開文件中。


## 三、學生在 GitHub 建立自己的 WebApp 副本

### 1. Fork 老師的 repository

1. 開啟範本 repository：

```text
https://github.com/chingmu-kuroro/taichung-rental-lightgbm
```

2. 按右上角 `Fork`。
3. Owner 選自己的 GitHub 帳號。
4. Repository name 可保留：

```text
taichung-rental-lightgbm
```

5. 按 `Create fork`。

完成後，學生會得到自己的 GitHub repository，例如：

```text
https://github.com/<GITHUB_USER>/taichung-rental-lightgbm
```

其中 `<GITHUB_USER>` 請替換成學生自己的 GitHub 帳號。


### 2. 從自己的 fork 開啟 Codespace

請務必在「自己的 fork」開 Codespace，不要直接在老師的 repository 開。

操作步驟：

1. 進入自己的 fork：

```text
https://github.com/<GITHUB_USER>/taichung-rental-lightgbm
```

2. 按綠色 `Code` 按鈕。
3. 切換到 `Codespaces` 頁籤。
4. 按 `Create codespace on main`。
5. 等待 Codespace 建立完成。

建立完成後，瀏覽器會開啟一個線上 VS Code 介面，左側可看到專案檔案，底部可開啟 Terminal。


## 四、Codespace 開啟後的第一次檢查

在 Codespace terminal 中執行：

```bash
pwd
git status
python --version
git lfs version
```

確認目前位於 repository 根目錄，且 Git 狀態正常。


### 1. 確認 Git LFS 大型檔案已下載

本專案的 `.gpkg` 與 `.joblib` 由 Git LFS 管理。請執行：

```bash
git lfs install
git lfs pull
git lfs ls-files
```

應該會看到類似下列檔案：

```text
Taichung_rental_houses_v4.gpkg
POIs/Taichung_youbikes.gpkg
artifacts/lightgbm_rent_model.joblib
artifacts/spatial_cache.joblib
```

如果 `.gpkg` 或 `.joblib` 檔案異常小，通常代表 Git LFS 沒有正確下載，請重新執行：

```bash
git lfs pull
```


### 2. 安裝 Python 套件

在 Codespace terminal 執行：

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

若安裝過程需要一些時間，請耐心等待。


## 五、在 Codespace 中本機測試 WebApp

在 Codespace terminal 執行：

```bash
export SOLARA_APP=app.py
export SOLARA_PRODUCTION=true
export APP_MAP_MAX_POINTS=15000

python -m uvicorn solara.server.starlette:app \
  --host 0.0.0.0 \
  --port 8765 \
  --proxy-headers \
  --forwarded-allow-ips="*"
```

注意：

- 在 Codespaces 中，`--host` 建議使用 `0.0.0.0`，這樣 GitHub 才能轉發 port。
- `APP_MAP_MAX_POINTS=15000` 表示地圖預設最多渲染 15,000 筆租屋點，避免地圖初次載入太慢。
- 若設定 `APP_MAP_MAX_POINTS=0`，代表不限制點數，會顯示全部租屋點。

啟動後，Codespaces 通常會提示有 port 8765 可開啟。也可以從下方 `PORTS` 面板找到 `8765`，按 `Open in Browser`。

請檢查：

1. 首頁能載入。
2. 【地圖預測】頁籤能開啟。
3. 地圖能顯示租屋點。
4. 可以在地圖上點選目標位置。
5. 右側可以輸入房屋物件屬性。
6. 按下「計算預測租金」後能得到結果。

測試結束後，可在 terminal 按：

```text
Ctrl + C
```

停止 Solara server。


## 六、在 HuggingFace 建立自己的 Space

學生可用 HuggingFace 網頁或 CLI 建立 Space。課堂上建議先用網頁建立，畫面比較直觀。


### 方法 A：使用 HuggingFace 網頁建立 Space

1. 登入 HuggingFace。
2. 點選右上角個人頭像或選單。
3. 選擇 `New Space`。
4. Owner 選自己的 HuggingFace 帳號。
5. Space name 例如：

```text
taichung-rental-lightgbm
```

6. SDK 選擇：

```text
Docker
```

7. Visibility 可選 `Public` 或 `Private`。
8. Hardware 可先選免費的 `CPU basic`。
9. 按下建立 Space。

建立後，學生會得到自己的 Space repository，例如：

```text
https://huggingface.co/spaces/<HF_USER>/<SPACE_NAME>
```


### 方法 B：在 Codespace 使用 HuggingFace CLI 建立 Space

先安裝 HuggingFace CLI：

```bash
python -m pip install --upgrade huggingface_hub
```

登入 HuggingFace：

```bash
hf auth login --add-to-git-credential
```

系統會要求貼上 HuggingFace token。請使用具有 write 權限的 token。

確認登入：

```bash
hf auth whoami
```

建立 Docker Space：

```bash
export HF_SPACE_ID="<HF_USER>/<SPACE_NAME>"

hf repos create "$HF_SPACE_ID" --repo-type=space --space-sdk=docker --exist-ok
```

範例：

```bash
export HF_SPACE_ID="student123/taichung-rental-lightgbm"

hf repos create "$HF_SPACE_ID" --repo-type=space --space-sdk=docker --exist-ok
```


## 七、從 Codespace 部署到 HuggingFace Space

課堂建議使用 `hf upload`，不要一開始就教學生用 `git push hf main`。原因是 HuggingFace Space 本身也是 Git repository，如果 Space 已經有初始 README commit，初學者容易遇到 Git history 合併、LFS、remote 衝突等問題。

`hf upload` 的概念比較單純：

```text
把目前 Codespace 專案資料夾中需要部署的檔案，上傳到指定 HuggingFace Space。
```


### 1. 確認已安裝與登入 HuggingFace CLI

```bash
python -m pip install --upgrade huggingface_hub
hf auth whoami
```

如果尚未登入：

```bash
hf auth login --add-to-git-credential
```


### 2. 設定 Space ID

在 terminal 設定：

```bash
export HF_SPACE_ID="<HF_USER>/<SPACE_NAME>"
```

範例：(雙引號內只能是小寫)

```bash
export HF_SPACE_ID="student123/taichung-rental-lightgbm"
```


### 3. 上傳部署檔案

在 repository 根目錄執行：

```bash
hf upload "$HF_SPACE_ID" . . \
  --repo-type=space \
  --commit-message "Deploy rental prediction WebApp" \
  --exclude ".git/**" \
  --exclude ".venv/**" \
  --exclude "__pycache__/**" \
  --exclude "**/__pycache__/**" \
  --exclude "*.pyc" \
  --exclude "**/*.pyc" \
  --exclude ".ipynb_checkpoints/**" \
  --exclude "*.ipynb"
```

上傳完成後，HuggingFace 會自動開始 build Space。

請開啟：

```text
https://huggingface.co/spaces/<HF_USER>/<SPACE_NAME>
```

或 App URL：

```text
https://<HF_USER>-<SPACE_NAME>.hf.space
```

實際 App URL 以 HuggingFace Space 頁面顯示為準。


## 八、學生修改後如何同步 GitHub 與 HuggingFace

這是本課程最重要的維護觀念：

```text
GitHub repository：保存開發版本與程式碼歷史。
HuggingFace Space：保存部署版本並執行 WebApp。
```

每次修改後，建議分成兩步：

1. 先 commit 並 push 到自己的 GitHub。
2. 再用 `hf upload` 部署到自己的 HuggingFace Space。


### 1. 更新自己的 GitHub repository

在 Codespace terminal 執行：

```bash
git status
git add .
git commit -m "Update WebApp"
git push origin main
```

如果 `git status` 顯示有不想保存的暫存檔，請不要直接 `git add .`，改用指定檔案：

```bash
git add src README.md requirements.txt
git commit -m "Update app code"
git push origin main
```


### 2. 更新自己的 HuggingFace Space

確認已安裝與登入 HuggingFace CLI，並已設定 Space ID之後：

```bash
hf upload "$HF_SPACE_ID" . . \
  --repo-type=space \
  --commit-message "Update deployed WebApp" \
  --exclude ".git/**" \
  --exclude ".venv/**" \
  --exclude "__pycache__/**" \
  --exclude "**/__pycache__/**" \
  --exclude "*.pyc" \
  --exclude "**/*.pyc" \
  --exclude ".ipynb_checkpoints/**" \
  --exclude "*.ipynb"
```


## 九、建議使用 Codespaces Secret 保存 HuggingFace token

如果不想每次開 Codespace 都重新登入 HuggingFace，可以使用 GitHub Codespaces secret。

### 1. 建立 Codespaces secret

在 GitHub：

1. 點右上角個人頭像。
2. 進入 `Settings`。
3. 找到 `Codespaces`。
4. 找到 `Codespaces secrets`。
5. 點 `New secret`。
6. Name 設為：

```text
HF_TOKEN
```

7. Value 貼上 HuggingFace write token。
8. Repository access 選擇自己的 `taichung-rental-lightgbm` repository。


### 2. 在 Codespace 使用 secret 登入

重新開啟或 rebuild Codespace 後，terminal 中應可讀取：

```bash
echo $HF_TOKEN
```

為避免 token 顯示在畫面上，正式操作時建議不要執行上面這行。

可直接登入：

```bash
hf auth login --token "$HF_TOKEN" --add-to-git-credential
hf auth whoami
```

注意：不要把 token 寫進 `.py`、`.md`、`.env` 後 commit。


## 十、WebApp 主要檔案與資料夾角色

```text
app.py
```

Solara WebApp 入口。HuggingFace Space 與 Codespace 本機測試都會從這個檔案啟動。

```text
src/config.py
```

專案設定檔。包含模型特徵欄位、POI 圖層設定、檔案路徑、座標系統、特徵說明文字等。

```text
src/model.py
```

模型訓練、載入、預測與特徵重要性整理。使用 LightGBM 預測租金，並計算 `Wy`、`W_pet_friendly` 等空間落後變數。

```text
src/spatial.py
```

空間運算邏輯。根據使用者在地圖上點選的目標坐標，計算至火車站、捷運站、ubike、公園、學校等區位特徵。

```text
src/map_view.py
```

leafmap 與 ipyleaflet 地圖元件。負責顯示 OpenStreetMap、租屋點位、目標標記與點選互動。

```text
src/ui.py
```

Solara 使用者介面。包含首頁、地圖預測頁籤、右側輸入表單、預測結果顯示。

```text
scripts/train_model.py
```

重新訓練 LightGBM 模型並輸出模型 artifact。

```text
scripts/build_spatial_cache.py
```

重新建立空間運算快取，讓 WebApp 在部署環境中能快速計算區位特徵。

```text
artifacts/
```

保存模型與空間快取。

```text
POIs/
```

保存各類 POI 與路網圖資，例如捷運站、ubike、學校、公園、商店、公車站等。

```text
Taichung_rental_houses_v4.gpkg
```

租屋點位與屬性原始資料。

```text
requirements.txt
```

Python 套件需求清單。

```text
Dockerfile
```

HuggingFace Docker Space 建置環境所需檔案。

```text
README.md
```

GitHub 與 HuggingFace Space 首頁說明，同時包含 HuggingFace Space metadata，例如：

```yaml
---
title: Taichung Rental LightGBM
emoji: 🏠
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
---
```

```text
.gitattributes
```

設定 `.gpkg` 與 `.joblib` 使用 Git LFS 管理。


## 十一、學生如何修改成自己的 WebApp

學生完成複製與部署後，可以逐步修改成自己的主題。


### 1. 更換資料檔案

目前主要資料檔是：

```text
Taichung_rental_houses_v4.gpkg
```

學生可使用兩種方式：

方法一：保留同樣檔名，直接用自己的資料覆蓋。

方法二：使用新的檔名，並修改 `src/config.py` 中的資料路徑。

若資料是大型檔案，請確認 Git LFS 有追蹤：

```bash
git lfs track "*.gpkg"
git lfs track "*.joblib"
git add .gitattributes
```

在 Codespace 中替換檔案的方法：

- 直接在 VS Code Explorer 拖曳上傳檔案。
- 使用 `wget` 或 `curl` 從雲端下載資料。
- 從 GitHub 或 HuggingFace Dataset 下載資料。


### 2. 修改特徵變數

主要修改位置：

```text
src/config.py
```

需要檢查：

- `BASE_X`
- `INTER_VARS`
- `SPATIAL_VARS`
- `FEATURE_COLUMNS`
- `FEATURE_DESCRIPTIONS`
- `USER_CONTINUOUS`
- `USER_BINARY`

如果新增或移除特徵，必須同步修改：

- 模型訓練資料欄位。
- WebApp 右側輸入表單。
- 特徵說明表格。
- 預測時的特徵組合邏輯。


### 3. 修改 POI 圖層與空間運算

主要修改位置：

```text
src/config.py
src/spatial.py
```

如果研究區不再是臺中，通常需要替換：

- 路網圖層。
- 行政區圖層。
- 捷運、火車站、ubike、公園、學校、商店、公車站、醫療診所等 POI 圖層。
- 核心區判定邏輯。

替換 POI 後，需重新建立空間快取：

```bash
python scripts/build_spatial_cache.py
```


### 4. 重新訓練模型

修改資料與特徵後，需要重新訓練：

```bash
python scripts/train_model.py
```

訓練完成後會更新：

```text
artifacts/lightgbm_rent_model.joblib
```

如果空間圖資也有修改，還需要更新：

```text
artifacts/spatial_cache.joblib
```


### 5. 修改介面文字與版面

主要修改位置：

```text
src/ui.py
src/map_view.py
```

可修改內容包括：

- WebApp 標題。
- 首頁說明文字。
- 模型指標顯示。
- 特徵說明表格。
- 地圖頁籤文字。
- 右側輸入欄位名稱。
- 預測結果顯示單位。


## 十二、課堂最短操作流程

以下是學生完成 fork 並開啟 Codespace 後，可直接依序執行的簡化版命令。

```bash
git lfs install
git lfs pull
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade huggingface_hub
hf auth login --add-to-git-credential
hf auth whoami
export HF_SPACE_ID="<HF_USER>/<SPACE_NAME>"
hf repos create "$HF_SPACE_ID" --repo-type=space --space-sdk=docker --exist-ok
hf upload "$HF_SPACE_ID" . . \
  --repo-type=space \
  --commit-message "Deploy rental prediction WebApp" \
  --exclude ".git/**" \
  --exclude ".venv/**" \
  --exclude "__pycache__/**" \
  --exclude "**/__pycache__/**" \
  --exclude "*.pyc" \
  --exclude "**/*.pyc" \
  --exclude ".ipynb_checkpoints/**" \
  --exclude "*.ipynb"
```


## 十三、常見問題

### 1. Codespace 開不起來或很慢

可能原因：

- GitHub Codespaces 額度不足。
- repository 較大，包含 Git LFS 檔案。
- Codespace machine 規格較低。

建議：

- 等待初始化完成。
- 確認不用的 Codespace 已停止或刪除。
- 課堂前先提醒學生建立 GitHub 帳號並確認 Codespaces 可用。


### 2. GPKG 或 joblib 檔案無法讀取

可能是 Git LFS 沒有拉下大型檔案。

請執行：

```bash
git lfs pull
git lfs ls-files
```


### 3. HuggingFace 登入失敗

請確認：

- token 有 write 權限。
- token 沒有過期。
- 使用的是 HuggingFace token，不是 GitHub token。
- 沒有把 `<HF_USER>` 寫成 GitHub 帳號。


### 4. Space build 失敗

請到 HuggingFace Space 頁面查看 `Logs`。

常見原因：

- `requirements.txt` 缺少套件。
- `Dockerfile` 指令錯誤。
- `README.md` metadata 錯誤。
- Space SDK 沒有設定為 Docker。
- `app_port` 不是 7860。
- `POIs/` 或 `artifacts/` 檔案缺漏。
- 大型檔案沒有完整上傳。


### 5. README metadata 錯誤

HuggingFace Space 的 `README.md` 最上方需要 YAML metadata，例如：

```yaml
---
title: Taichung Rental LightGBM
emoji: 🏠
colorFrom: green
colorTo: red
sdk: docker
app_port: 7860
---
```

`emoji` 必須是真正的圖示字元，不能寫成 `home` 這類文字。


### 6. 修改後 GitHub 有更新，但 HuggingFace Space 沒更新

這是正常的。GitHub 與 HuggingFace Space 是兩個不同平台。

學生需要再執行一次：

```bash
hf upload "$HF_SPACE_ID" . . --repo-type=space --commit-message "Update deployed WebApp"
```

如果使用多行版指令，請保留第八節的 exclude 設定。


### 7. Codespace 中 Solara 啟動後看不到網頁

請確認：

- 使用 `--host=0.0.0.0`。
- port 使用 `8765` 或其他未被占用的 port。
- 到 Codespaces 下方 `PORTS` 面板開啟 forwarded port。

建議指令：

```bash
solara run app.py --host=0.0.0.0 --port=8765
```


## 十四、進階：雙 remote 維護方式

熟悉 Git 後，可以把 HuggingFace Space 也設為 Git remote：

```bash
git remote add hf https://huggingface.co/spaces/<HF_USER>/<SPACE_NAME>
```

日常開發推送到 GitHub：

```bash
git add .
git commit -m "Update WebApp"
git push origin main
```

部署時推送到 HuggingFace：

```bash
git push hf main
```

不過，這不是本課堂初學者的主要建議流程。若 Space repository 已經有初始 commit，直接 `git push hf main` 可能被拒絕，需要處理 remote history。初學者請優先使用：

```bash
hf upload
```


## 十五、建議的課堂作業檢核表

學生完成後，至少應繳交：

1. 自己的 GitHub repository URL。
2. 自己的 HuggingFace Space URL。
3. Codespace 中成功執行 WebApp 的截圖。
4. HuggingFace Space 成功執行 WebApp 的截圖。
5. 說明自己修改了哪些資料、特徵或介面文字。
6. 說明是否重新訓練模型或重建空間快取。


## 十六、教師可進一步簡化的方向

若希望學生開啟 Codespace 後幾乎不用安裝套件，可以在 repository 中額外加入：

```text
.devcontainer/devcontainer.json
```

用 devcontainer 預先定義 Python、Git LFS、系統套件與 pip 安裝流程。這可以讓學生的 Codespace 更一致，但也會增加教師維護 devcontainer 的工作。

若希望部署指令更短，也可以新增：

```text
scripts/deploy_to_hf.sh
```

讓學生只要設定：

```bash
export HF_SPACE_ID="<HF_USER>/<SPACE_NAME>"
```

再執行：

```bash
bash scripts/deploy_to_hf.sh
```

即可部署。


## 十七、參考資料

- GitHub Codespaces documentation：`https://docs.github.com/en/codespaces`
- GitHub creating a codespace documentation：`https://docs.github.com/en/codespaces/developing-in-a-codespace/creating-a-codespace-for-a-repository`
- GitHub Codespaces secrets documentation：`https://docs.github.com/en/codespaces/managing-your-codespaces/managing-your-account-specific-secrets-for-github-codespaces`
- GitHub fork repository documentation：`https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo`
- HuggingFace Spaces overview：`https://huggingface.co/docs/hub/spaces-overview`
- HuggingFace Docker Spaces documentation：`https://huggingface.co/docs/hub/spaces-sdks-docker`
- HuggingFace CLI documentation：`https://huggingface.co/docs/huggingface_hub/en/guides/cli`
- Git LFS documentation：`https://github.com/git-lfs/git-lfs`
