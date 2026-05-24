# 學生複製與部署臺中租金預測 WebApp 操作說明

版本日期：2026-05-24

本文件提供課堂使用的完整流程，說明學生如何在自己的 GitHub 與 HuggingFace 帳號下複製本 WebApp，部署到自己的 HuggingFace Space，並在理解程式架構後更換成自己的資料、特徵變數與模型。

範本 GitHub repository：

```text
https://github.com/chingmu-kuroro/taichung-rental-lightgbm
```

範本 HuggingFace Space：

```text
https://huggingface.co/spaces/Chingmu/taichung-rental-lightgbm
```


## 一、整體工作流程

學生的建議流程如下：

1. 在 GitHub 上 fork 老師提供的 WebApp repository。
2. 將自己的 fork clone 到本機。
3. 安裝 Git LFS，下載大型圖資與模型檔案。
4. 在本機測試 WebApp 是否能執行。
5. 在 HuggingFace 建立自己的 Space。
6. 使用 HuggingFace CLI 將 WebApp 上傳到自己的 Space。
7. 確認 HuggingFace Space 建置成功並能開啟。
8. 之後再逐步更換自己的資料、特徵變數、模型與介面內容。

課堂建議採用「GitHub fork + HuggingFace CLI upload」作為主要流程。這個方法最穩定，能避免初學者遇到 GitHub repository 與 HuggingFace Space repository 之間的 Git history 合併問題。


## 二、學生需要先準備的帳號與工具

### 1. 帳號

每位學生需要：

- 一個 GitHub 帳號。
- 一個 HuggingFace 帳號。

建議學生先登入以下網站：

```text
https://github.com
https://huggingface.co
```

### 2. 本機軟體

建議安裝：

- Git
- Git LFS
- Python 3.11 或 Anaconda
- HuggingFace CLI

在 PowerShell 檢查：

```powershell
git --version
git lfs version
python --version
```

若尚未安裝 Git LFS，可到 Git LFS 官方頁面下載安裝，或依作業系統使用套件管理工具安裝。


## 三、在 GitHub 建立自己的 WebApp 副本

### 1. Fork 老師的 GitHub repository

1. 開啟範本 repository：

```text
https://github.com/chingmu-kuroro/taichung-rental-lightgbm
```

2. 按右上角的 `Fork`。
3. Owner 選擇自己的 GitHub 帳號。
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


### 2. Clone 自己的 fork 到本機

在 PowerShell 選擇一個工作資料夾，例如：

```powershell
cd D:\CourseWork
```

clone 自己的 repository：

```powershell
git clone https://github.com/<GITHUB_USER>/taichung-rental-lightgbm.git
cd taichung-rental-lightgbm
```

設定 Git LFS：

```powershell
git lfs install
git lfs pull
```

檢查 LFS 檔案：

```powershell
git lfs ls-files
```

應該會看到 `.gpkg` 與 `.joblib` 檔案由 Git LFS 管理，例如：

```text
Taichung_rental_houses_v4.gpkg
POIs/Taichung_youbikes.gpkg
artifacts/lightgbm_rent_model.joblib
artifacts/spatial_cache.joblib
```


## 四、在本機執行 WebApp

### 1. 建立 Python 環境

若使用 Python venv：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若使用 conda：

```powershell
conda create -n rental-webapp python=3.11
conda activate rental-webapp
python -m pip install --upgrade pip
pip install -r requirements.txt
```


### 2. 執行 Solara WebApp

```powershell
solara run app.py --host=127.0.0.1 --port=8765
```

開啟瀏覽器：

```text
http://127.0.0.1:8765
```

如果地圖點位太多導致載入較慢，可設定地圖最多顯示點數：

```powershell
$env:APP_MAP_MAX_POINTS="15000"
solara run app.py --host=127.0.0.1 --port=8765
```

若設定：

```powershell
$env:APP_MAP_MAX_POINTS="0"
```

代表不限制點數，會顯示全部租屋點位。


## 五、WebApp 主要檔案與資料夾角色

```text
app.py
```

Solara WebApp 入口。HuggingFace Space 與本機執行都會從這個檔案啟動。

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
artifacts/lightgbm_rent_model.joblib
artifacts/spatial_cache.joblib
```

WebApp 預測時會載入的模型與空間快取檔案。

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
sdk: docker
app_port: 7860
```

```text
.gitattributes
```

設定 `.gpkg` 與 `.joblib` 使用 Git LFS 管理。


## 六、建立自己的 HuggingFace Space

### 方法 A：使用 HuggingFace 網頁建立 Space

1. 登入 HuggingFace。
2. 點選右上角個人頭像或選單。
3. 選擇 `New Space`。
4. Owner 選擇自己的 HuggingFace 帳號。
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
9. 建立 Space。

Space 建立後，會有一個 repository URL，例如：

```text
https://huggingface.co/spaces/<HF_USER>/<SPACE_NAME>
```


### 方法 B：使用 HuggingFace CLI 建立 Space

先安裝 HuggingFace CLI：

```powershell
python -m pip install --upgrade huggingface_hub
```

登入：

```powershell
hf auth login --add-to-git-credential
```

確認登入：

```powershell
hf auth whoami
```

建立 Docker Space：

```powershell
hf repos create spaces/<HF_USER>/<SPACE_NAME> --repo-type=space --space-sdk=docker --exist-ok
```

其中：

- `<HF_USER>`：學生自己的 HuggingFace 帳號。
- `<SPACE_NAME>`：學生自己的 Space 名稱。


## 七、部署到自己的 HuggingFace Space

### 課堂建議方法：使用 `hf upload`

這個方法最適合初學者，因為不需要處理 GitHub repository 與 HuggingFace Space repository 的 history 合併問題。

在專案資料夾中執行：

```powershell
hf upload <HF_USER>/<SPACE_NAME> . . --repo-type=space --commit-message "Deploy rental prediction WebApp" --exclude ".git/**" --exclude ".venv/**" --exclude "__pycache__/**" --exclude "**/__pycache__/**" --exclude "*.pyc" --exclude "**/*.pyc" --exclude ".ipynb_checkpoints/**" --exclude "*.ipynb"
```

範例：

```powershell
hf upload student123/taichung-rental-lightgbm . . --repo-type=space --commit-message "Deploy rental prediction WebApp" --exclude ".git/**" --exclude ".venv/**" --exclude "__pycache__/**" --exclude "**/__pycache__/**" --exclude "*.pyc" --exclude "**/*.pyc" --exclude ".ipynb_checkpoints/**" --exclude "*.ipynb"
```

上傳完成後，HuggingFace 會自動開始 build Space。

學生可開啟：

```text
https://huggingface.co/spaces/<HF_USER>/<SPACE_NAME>
```

或 App 網址：

```text
https://<HF_USER>-<SPACE_NAME>.hf.space
```

實際 App 網址以 HuggingFace Space 頁面顯示為準。


## 八、進階維護方法：GitHub 與 HuggingFace 雙 remote

如果學生後續熟悉 Git，可以採用雙 remote 維護：

- `origin`：自己的 GitHub repository。
- `hf`：自己的 HuggingFace Space repository。

設定 HuggingFace remote：

```powershell
git remote add hf https://huggingface.co/spaces/<HF_USER>/<SPACE_NAME>
```

日常開發推送到 GitHub：

```powershell
git add .
git commit -m "Update WebApp"
git push origin main
```

部署時推送到 HuggingFace：

```powershell
git push hf main
```

注意：如果 Space 是用 HuggingFace 網頁建立，Space repository 可能已經有初始 README commit，直接 `git push hf main` 可能被拒絕。初學者建議先使用第七節的 `hf upload`。等熟悉 Git history 後，再處理雙 remote 合併。


## 九、學生如何修改成自己的 WebApp

學生完成複製與部署後，可依序修改下列部分。

### 1. 更換租屋或研究資料

目前主要資料檔是：

```text
Taichung_rental_houses_v4.gpkg
```

若學生要使用自己的資料，有兩種方式：

方法一：保留同樣檔名，直接用新資料覆蓋。

方法二：使用新的檔名，並修改 `src/config.py` 中的資料路徑設定。

資料至少需要：

- 幾何欄位。
- 可轉換至模型使用座標系統的 CRS。
- 模型訓練所需的目標變數。
- 模型特徵欄位。


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


### 3. 修改空間運算與 POI 圖層

主要修改位置：

```text
src/config.py
src/spatial.py
```

如果學生研究區不再是臺中，通常需要替換：

- 路網圖層。
- 行政區圖層。
- 捷運、火車站、ubike、公園、學校、商店、公車站、醫療診所等 POI 圖層。
- 核心區判定邏輯。

替換 POI 後，需重新建立空間快取：

```powershell
python scripts\build_spatial_cache.py
```


### 4. 重新訓練模型

修改資料與特徵後，需要重新訓練：

```powershell
python scripts\train_model.py
```

訓練完成後會更新：

```text
artifacts/lightgbm_rent_model.joblib
```

如果空間圖資也有改，還需要更新：

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


### 6. 本機測試

每次改完後，先在本機測試：

```powershell
solara run app.py --host=127.0.0.1 --port=8765
```

確認：

- 首頁能載入。
- 地圖預測頁籤能載入。
- 地圖點位能顯示。
- 使用者能點選目標位置。
- 按下預測按鈕後能得到結果。
- HuggingFace 部署所需檔案沒有缺漏。


### 7. 更新 GitHub 與 HuggingFace

改完後推送到自己的 GitHub：

```powershell
git add .
git commit -m "Update model and app"
git push origin main
```

再部署到自己的 HuggingFace Space：

```powershell
hf upload <HF_USER>/<SPACE_NAME> . . --repo-type=space --commit-message "Update deployed WebApp" --exclude ".git/**" --exclude ".venv/**" --exclude "__pycache__/**" --exclude "**/__pycache__/**" --exclude "*.pyc" --exclude "**/*.pyc" --exclude ".ipynb_checkpoints/**" --exclude "*.ipynb"
```


## 十、常見問題

### 1. GitHub clone 後資料檔案很小，或 GPKG 無法讀取

可能是 Git LFS 檔案沒有下載完整。

請執行：

```powershell
git lfs install
git lfs pull
```


### 2. HuggingFace Space build 失敗

請到 Space 頁面的 `Logs` 檢查錯誤。

常見原因：

- `requirements.txt` 缺少套件。
- `Dockerfile` 指令錯誤。
- `README.md` metadata 錯誤。
- Space SDK 未設定為 Docker。
- `app_port` 不是 7860。
- `POIs/` 或 `artifacts/` 檔案缺漏。
- 大型檔案沒有完整上傳。


### 3. README metadata 錯誤

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


### 4. Space 可以開啟，但地圖或預測不能運作

請檢查：

- `Taichung_rental_houses_v4.gpkg` 是否存在。
- `POIs/` 內所有必要 `.gpkg` 是否存在。
- `artifacts/lightgbm_rent_model.joblib` 是否存在。
- `artifacts/spatial_cache.joblib` 是否存在。
- `src/config.py` 的檔案路徑是否正確。
- HuggingFace Logs 是否有 Python traceback。


### 5. Git push 要求密碼但失敗

GitHub 不再接受一般帳號密碼作為 Git push 認證。請使用：

- Git Credential Manager 的瀏覽器登入。
- GitHub personal access token。
- SSH key。

課堂上若只需要部署到 HuggingFace，可先使用 `hf upload`，不一定要處理 GitHub CLI。


## 十一、建議的課堂作業檢核表

學生完成後，至少應能提交以下成果：

1. 自己的 GitHub repository URL。
2. 自己的 HuggingFace Space URL。
3. 本機成功執行 WebApp 的截圖。
4. HuggingFace Space 成功執行 WebApp 的截圖。
5. 說明自己修改了哪些資料、特徵或介面文字。
6. 說明重新訓練模型或重建空間快取的步驟。


## 十二、後續改作建議

學生可以依自己的研究主題進行改作，例如：

- 更換研究區。
- 更換租金以外的預測目標。
- 更換 POI 類型。
- 新增不同距離尺度的環域特徵。
- 改用不同模型。
- 改成房價、店租、民宿價格、土地價格或其他空間預測 WebApp。

改作時請優先掌握三個核心：

1. `src/config.py` 控制資料、特徵與欄位。
2. `src/model.py` 控制模型訓練與預測。
3. `src/spatial.py` 控制地圖點選後的空間運算。

只要理解這三個檔案，學生就能逐步把範本改造成自己的 WebApp。


## 十三、參考資料

- GitHub fork repository documentation：`https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo`
- GitHub git clone guide：`https://github.com/git-guides/git-clone`
- HuggingFace Spaces overview：`https://huggingface.co/docs/hub/spaces-overview`
- HuggingFace CLI documentation：`https://huggingface.co/docs/huggingface_hub/en/guides/cli`
- Git LFS project documentation：`https://github.com/git-lfs/git-lfs`

