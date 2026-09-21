$ErrorActionPreference = "Stop"
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force -Path data,data/genomes,data/blast,models,results,figures,logs | Out-Null

$prov = @()
function Reg($name,$url,$path,$note){
  $fi = Get-Item $path
  $sha = (Get-FileHash -Path $path -Algorithm SHA256).Hash
  $script:prov += [ordered]@{ name=$name; url=$url; local_path=$path; bytes=$fi.Length; sha256=$sha; retrieved_utc=(Get-Date).ToUniversalTime().ToString("o"); note=$note }
  "OK  {0,-20} {1,12:N0} B  {2}" -f $name,$fi.Length,$sha.Substring(0,12)
}

# ---------- 1. Genomes: 7 S. boulardii assemblies ----------
$asm = [ordered]@{
  "Sb_unique28"    = "GCA_001413975.1"
  "Sb_PY0001"      = "GCA_024732265.1"
  "Sb_KCTC13826BP" = "GCA_026225675.1"
  "Sb_EDRL"        = "GCA_000442675.2"
  "Sb_CLPY01"      = "GCA_021216695.1"
  "Sb_ATCC_MYA797" = "GCA_001625055.1"
  "Sb_strain17"    = "GCA_000734875.3"
}
foreach($k in $asm.Keys){
  $acc = $asm[$k]
  $zip = "data/genomes/$k.zip"
  if(-not (Test-Path $zip)){
    $u = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/$acc/download?include_annotation_type=GENOME_FASTA"
    Invoke-WebRequest -Uri $u -OutFile $zip -TimeoutSec 900
  }
  Reg $k ("NCBI Datasets v2 " + $acc) $zip "S. boulardii assembly"
}

# S288C reference: genome + proteins + GFF
$zipS = "data/genomes/Sc_S288C.zip"
if(-not (Test-Path $zipS)){
  $u = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCF_000146045.2/download?include_annotation_type=GENOME_FASTA&include_annotation_type=PROT_FASTA&include_annotation_type=GENOME_GFF"
  Invoke-WebRequest -Uri $u -OutFile $zipS -TimeoutSec 1200
}
Reg "Sc_S288C" "NCBI Datasets v2 GCF_000146045.2" $zipS "S288C R64: genome+protein+GFF"

# ---------- 2. Yeast9 GEM ----------
$rel = Invoke-RestMethod -Uri "https://api.github.com/repos/SysBioChalmers/yeast-GEM/releases/latest" -Headers @{'User-Agent'='research-script'} -TimeoutSec 120
$tag = $rel.tag_name
"yeast-GEM release tag: $tag"
$gem = "data/yeast-GEM.xml"
if(-not (Test-Path $gem)){
  Invoke-WebRequest -Uri "https://raw.githubusercontent.com/SysBioChalmers/yeast-GEM/$tag/model/yeast-GEM.xml" -OutFile $gem -TimeoutSec 900
}
Reg "yeast-GEM.xml" "https://raw.githubusercontent.com/SysBioChalmers/yeast-GEM/$tag/model/yeast-GEM.xml" $gem "Yeast9 consensus GEM, release $tag"

# ---------- 3. BLAST+ Windows binaries ----------
$idx = Invoke-WebRequest -Uri "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/" -UseBasicParsing -TimeoutSec 180
$m = [regex]::Matches($idx.Content, 'ncbi-blast-[0-9.]+\+-x64-win64\.tar\.gz')
if($m.Count -gt 0){
  $fn = $m[0].Value
  $bt = "data/blast/$fn"
  if(-not (Test-Path $bt)){
    Invoke-WebRequest -Uri "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/$fn" -OutFile $bt -TimeoutSec 1800
  }
  Reg "blast_plus" "https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/$fn" $bt "NCBI BLAST+ win64"
  tar -xzf $bt -C data/blast
  $tb = (Get-ChildItem -Path data/blast -Recurse -Filter "tblastn.exe" | Select-Object -First 1)
  "BLAST tblastn: " + $(if($tb){$tb.FullName}else{"NOT_FOUND"})
} else { "BLAST_INDEX_NO_MATCH" }

# ---------- 4. Unzip genomes ----------
foreach($z in (Get-ChildItem data/genomes -Filter *.zip)){
  $dest = "data/genomes/" + $z.BaseName
  if(-not (Test-Path $dest)){ Expand-Archive -Path $z.FullName -DestinationPath $dest -Force }
}
"FNA files: " + (Get-ChildItem data/genomes -Recurse -Include *.fna,*.faa | Measure-Object).Count

$prov | ConvertTo-Json -Depth 5 | Out-File -Encoding utf8 data/PROVENANCE.json
"WROTE data/PROVENANCE.json entries=" + $prov.Count
