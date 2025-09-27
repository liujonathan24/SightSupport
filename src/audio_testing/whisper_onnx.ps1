# download_whisper_onnx.ps1
$models = @("tiny.en", "base.en", "small.en", "medium.en")
$baseUrl = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
$modelDir = "whisper_onnx_models"

New-Item -ItemType Directory -Force -Path $modelDir | Out-Null

foreach ($model in $models) {
    $tarball = "$baseUrl/sherpa-onnx-whisper-$model.tar.bz2"
    $outFile = "$modelDir\$model.tar.bz2"

    Write-Host "Downloading $model ..."
    Invoke-WebRequest -Uri $tarball -OutFile $outFile

    Write-Host "Extracting $model ..."
    tar -xjf $outFile -C $modelDir
}
Write-Host "Done. Models are in $modelDir"
