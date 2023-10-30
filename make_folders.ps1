# JSONファイルを読み込む
$jsonData = Get-Content -Path 'make_folders.json' | ConvertFrom-Json

# ルートディレクトリを取得
$rootDir = $jsonData.root_directory

# ルートディレクトリが存在しない場合は作成
if (-not (Test-Path $rootDir)) {
    New-Item -Path $rootDir -ItemType Directory
}

# JSONデータに基づいてサブフォルダを作成
foreach ($folder in $jsonData.folders) {
    $folderPath = Join-Path -Path $rootDir -ChildPath $folder
    if (-not (Test-Path $folderPath)) {
        New-Item -Path $folderPath -ItemType Directory
    }
}
