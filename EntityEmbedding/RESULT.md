# EntityEmbedding 参数调整

## Embedding Size | architecture 32-16-8
 
|Embedding size|Full Fold AUC|Public AUC|Private AUC|
|------|------|------|------|
|log2(n)|0.79877|0.80681||
|2*log2(n)|0.80017|0.80749||
|4*log2(n)|0.80027|0.80755||
|6*log2(n)|0.79999|0.80761||
|8*log2(n)|0.79983|0.80735||

## Architecture | FIX Embedding Size 4*log2(n)

|Architecture|Full Fold AUC|Public AUC|Private AUC|
|------|------|------|------|
|8-8-8|0.80036|0.80751||
|16-16-16|0.80050|0.80753||
|32-32-32|0.80035|0.80745||
|64-64-64|0.79978|0.80752||
|128-128-128|0.79922|0.80748||

## Embedding Size | FIX Architecture 1

|Embedding size|Full Fold AUC|Public AUC|Private AUC|
|------|------|------|------|
|2*log2(n)|0.80122|0.80702||
|4*log2(n)|0.80191|0.80738||
|6*log2(n)|0.80185|0.80765||
|min(50, num // 2)|0.80026|0.80549||

## Activation | FIX Embedding Size 4*log2(n) | Architecture 1

|Activation|Full Fold AUC|Public AUC|Private AUC|
|------|------|------|------|
|sigmoid|0.80267|0.80711||
|tanh|0.80194|0.80734|