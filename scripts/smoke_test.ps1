$ErrorActionPreference = "Continue"
$ProgressPreference = 'SilentlyContinue'
$e = "C:\Users\lenovo\.claude-science\conda\envs\sbdt"
$env:PATH = "$e;$e\Library\mingw-w64\bin;$e\Library\usr\bin;$e\Library\bin;$e\Scripts;$e\bin;" + $env:PATH
$env:PYTHONUNBUFFERED = "1"

$job = Start-Job -ScriptBlock {
  param($py, $cwd, $path)
  $env:PATH = $path
  Set-Location $cwd
  & $py -m uvicorn app.main:app --port 8137 --log-level warning 2>&1
} -ArgumentList "$e\python.exe", (Get-Location).Path, $env:PATH

$ok = $false
for ($i = 0; $i -lt 60; $i++) {
  Start-Sleep -Seconds 2
  try { $h = Invoke-RestMethod -Uri "http://127.0.0.1:8137/health" -TimeoutSec 5; $ok = $true; break } catch {}
}
if (-not $ok) { "SERVER_FAILED"; Receive-Job $job | Select-Object -Last 30; Stop-Job $job; Remove-Job $job; exit 1 }
"health: " + ($h | ConvertTo-Json -Compress -Depth 3)

$results = @()
function Hit($name, $method, $url, $body) {
  $t0 = Get-Date
  try {
    if ($method -eq "GET") { $r = Invoke-RestMethod -Uri $url -TimeoutSec 180 }
    else { $r = Invoke-RestMethod -Uri $url -Method Post -ContentType "application/json" -Body $body -TimeoutSec 180 }
    $ms = [math]::Round(((Get-Date) - $t0).TotalMilliseconds)
    $j = ($r | ConvertTo-Json -Depth 6 -Compress)
    $script:results += [pscustomobject]@{ endpoint = $name; status = "ok"; ms = $ms; bytes = $j.Length; sample = $j.Substring(0, [math]::Min(150, $j.Length)) }
    "{0,-26} ok  {1,6} ms  {2,8} B" -f $name, $ms, $j.Length
  } catch {
    $script:results += [pscustomobject]@{ endpoint = $name; status = "FAIL"; ms = 0; bytes = 0; sample = $_.Exception.Message }
    "{0,-26} FAIL  {1}" -f $name, $_.Exception.Message
  }
}

$B = "http://127.0.0.1:8137"
Hit "GET /genome"              GET  "$B/genome"
Hit "GET /genome?gene=HXT9"    GET  "$B/genome?gene=HXT9"
Hit "GET /introgression"       GET  "$B/introgression"
Hit "GET /model_summary"       GET  "$B/model_summary"
Hit "GET /phenotypes"          GET  "$B/phenotypes"
Hit "GET /essentiality"        GET  "$B/essentiality"
Hit "GET /zone/ileum"          GET  "$B/zone/ileum"
Hit "GET /zone/stomach"        GET  "$B/zone/stomach"
Hit "GET /zone_contrasts"      GET  "$B/zone_contrasts"
Hit "GET /ph_response"         GET  "$B/ph_response"
Hit "GET /kinetics/posterior"  GET  "$B/kinetics/posterior"
Hit "GET /surrogate/metrics"   GET  "$B/surrogate/metrics"
Hit "GET /competition/map"     GET  "$B/competition/map"
Hit "POST /predict"            POST "$B/predict"     '{"glucose":2.5,"oxygen":3.0,"ammonium":5.0,"phosphate":1.0,"uncertainty":true}'
Hit "POST /kinetics"           POST "$B/kinetics"    '{"use_posterior":true,"t_end":8}'
Hit "POST /competition"        POST "$B/competition" '{"k_ox":0.8,"D":0.1,"S_in":25,"k_prot":1.2,"t_end":200}'
Hit "POST /query"              POST "$B/query"       '{"question":"which genes are absent in this strain?"}'
Hit "POST /fba glucose"        POST "$B/fba"         '{"carbon_source":"glucose","carbon_uptake":1.65,"oxygen":2.0,"model":"strain","top_n":10}'
Hit "POST /fba trehalose"      POST "$B/fba"         '{"carbon_source":"trehalose","carbon_uptake":10,"oxygen":20,"model":"strain","top_n":5}'
Hit "GET / (dashboard)"        GET  "$B/"

$results | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 results/api_smoke_test.json
"PASS: " + (($results | Where-Object { $_.status -eq 'ok' }).Count) + " / " + $results.Count

Stop-Job $job; Remove-Job $job -Force
