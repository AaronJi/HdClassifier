param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('train', 'infer')]
    [string]$Stage
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$python = '../../anaconda3/python.exe'
$env:PYTHONPATH = "$project/.runtime/llamafactory-deps;$project/vendor/LlamaFactory/src"
$env:HF_HOME = "$project/.runtime/huggingface"
$env:HF_HUB_OFFLINE = '1'
$env:HF_DATASETS_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:USE_V1 = '0'

Push-Location -LiteralPath $project
try {
    if ($Stage -eq 'train') {
        & $python -u -B -X utf8 -m llamafactory.cli train 'configs/wos_llama32_3b_train.yaml'
    } else {
        & $python -u -B -X utf8 'scripts/infer_wos.py' 'configs/wos_llama32_3b_infer.yaml'
    }
    $stageExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $stageExitCode
