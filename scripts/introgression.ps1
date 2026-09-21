$ErrorActionPreference="Stop"; $ProgressPreference='SilentlyContinue'
$zip="data/genomes/Zb_CLIB213.zip"
if(-not (Test-Path $zip)){
  Invoke-WebRequest -Uri "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/GCA_000442885.1/download?include_annotation_type=GENOME_FASTA&include_annotation_type=PROT_FASTA" -OutFile $zip -TimeoutSec 1200
}
if(-not (Test-Path "data/genomes/Zb_CLIB213")){ Expand-Archive -Path $zip -DestinationPath "data/genomes/Zb_CLIB213" -Force }
$faa=(Get-ChildItem data/genomes/Zb_CLIB213 -Recurse -Filter *.faa | Select-Object -First 1).FullName
"Zb proteome: $faa  (" + ((Select-String -Path $faa -Pattern '^>' -AllMatches).Count) + " proteins)"
$bin="data/blast/ncbi-blast-2.17.0+/bin"
$fields="qseqid sseqid pident length qlen qstart qend sstart send evalue bitscore"
foreach($t in @("Sb_unique28","Sb_PY0001","Sc_S288C")){
  $out="results/introgression_$t.tsv"
  if(-not (Test-Path $out)){
    & "$bin/tblastn.exe" -query $faa -db "data/blastwork/$t" -outfmt "6 $fields" -evalue 1e-10 -max_target_seqs 5 -num_threads 6 -seg no -out $out
  }
  "$t -> " + (Get-Content $out | Measure-Object -Line).Lines + " HSPs"
}
"INTROGRESSION_BLAST_DONE"
