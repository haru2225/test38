# test38: スパコン向け粘土CGデノイザー

sigma条件付きNequIPの学習と、reverse-SDE / DDIMによる構造生成。
`test38.py` は単独で実行でき、DM2やtest37のチェックアウトは不要です。
必要な外部Pythonパッケージはコンテナ定義に記載しています。

## スパコンで実行（PBS）

```bash
git clone https://github.com/haru2225/test38.git
cd test38
# 施設で必要なSingularity / Apptainerモジュールを先にロードしてください。
singularity build --fakeroot test38.sif Singularity.test38.def
qsub -P <ProjectGroup_ID> run_test38.pbs
```

`<ProjectGroup_ID>` は自分の課題番号に置き換えます。
Apptainer環境ではビルドコマンドの `singularity` を `apptainer` に置き換えてください。
ジョブスクリプトはApptainerを自動検出します。
ビルドにはネットワークとfakeroot対応の環境が必要です。施設がビルドを許可する
ノードで事前に作成するか、別環境で作成したSIFを転送してください。

既定は **sg8・GPU 1台・CPU 8個・32 GB・20時間**、学習時間予算19.5時間です。
キュー・資源指定は施設に合わせて変更してください。単一GPU実行です。
CUDA 12.4対応のホストGPUドライバーが必要です。
既定では `test37` と同じ `input/test36-dataset`（10,000フレーム）を使い、
30,000更新をゼロから学習します。`positions.npy` はGitHubのファイルサイズ制限を
避けるため2分割して保存し、PBSスクリプトが実行時に結合します。
結果は `results/train/` に保存されます。

1292サイトのグラフでは、sigma条件付きモデルのGPUメモリ使用量が大きくなるため、
PBSの既定バッチサイズは2です。メモリが足りない場合は `BATCH_SIZE=1` に下げてください。

```bash
# 学習終了後に構造生成（結果: results/generated/）
qsub -P <ProjectGroup_ID> -v STAGE=generate run_test38.pbs

# 時間切れなどで中断した学習・生成を再開
qsub -P <ProjectGroup_ID> -v RESUME=1 run_test38.pbs
qsub -P <ProjectGroup_ID> -v STAGE=generate,RESUME=1 run_test38.pbs

# 学習更新数とバッチサイズを指定
qsub -P <ProjectGroup_ID> -v UPDATES=30000,BATCH_SIZE=2 run_test38.pbs
```

再開時はデータ・デバイス・学習/生成設定を初回と同じにしてください。
学習の `UPDATES` は総更新数で、再開時に延長できます。
正常終了は終了コード0、中断してチェックポイントを保存した場合は75です。
生成は学習完了を確認してから投入してください。同じ出力先への同時投入は避けてください。
新しい実験では `TRAIN_DIR` / `GENERATED_DIR` に新しいディレクトリを指定します。

## 自分のデータ・既存モデルを使用

`DATASET_PATH` にtest36/37で準備したデータセット（`metadata.json`,
`positions.npy`, `cells.npy`）を指定できます。読み込み時にSHA-256を検証します。
同梱データはtest37リポジトリの `input/test36-dataset` と同じ10,000フレームの
データです。本番データに置き換える場合も同じ形式を使用してください。
データセット作成コマンドはこのリポジトリには含みません。

既存のtest36/37モデルから初期化する場合は、チェックポイントをコピーして
`WARM_START` に指定してください。モデル構成が一致する必要があります。
チェックポイント自体は同梱していません。

```bash
qsub -P <ProjectGroup_ID> \
  -v DATASET_PATH=input/my-dataset,WARM_START=input/checkpoint.pt \
  run_test38.pbs
```

ウォームスタート学習の再開でも同じ `WARM_START` を指定してください。
データと出力は原則リポジトリ内に配置します。外部ストレージを使う場合は
絶対パスと `EXTRA_BIND=/scratch:/scratch` などの追加マウントを指定してください。
既存コンテナは `SIF_IMAGE`、ランタイムは `CONTAINER_RUNTIME` で変更できます。
その他の設定変数は `run_test38.pbs` に記載しています。

## Pythonで直接実行

依存パッケージがインストールされた環境では以下でも実行できます。

```bash
python test38.py train --dataset input/test36-dataset --output results/train --device cuda
python test38.py generate --checkpoint results/train/checkpoint.pt \
  --output results/generated --device cuda
python test38.py train --help
python test38.py generate --help
```

`--device cpu` でCPU実行も可能です。生成結果は `positions.npy`,
`generation.json`, `generation_restart.pt`, `final.extxyz` に保存されます。
途中の軌跡は `generation.json` の `valid_frames` までが有効です。

## 途中チェックポイントの確認

学習が30,000更新に達する前でも、保存済みの `checkpoint.pt` は生成に使えます。
別の出力先へ生成し、参照データとの構造診断を実行してください。

```bash
python test38.py generate \
  --checkpoint results/train/checkpoint.pt \
  --output results/generated-intermediate \
  --device cuda --time-budget-hours 19.5

python analyze_test38.py \
  --reference input/test36-dataset \
  --checkpoint results/train/checkpoint.pt \
  --generated results/generated-intermediate \
  --output results/generated-intermediate-analysis.json
```

解析はNaN、最小原子間距離、フレーム間の変位、参照データとのRDF差を記録します。
途中生成は `generation_complete=false` として報告され、未完了のチェックポイントである
ことを示します。`status=pass` は構造上の簡易検査を通った意味で、平衡性や物理的妥当性を
保証しません。

CGMD軌道を別途作成済みなら、`md.extxyz` を追加指定して同じ検査を行えます。

```bash
python analyze_test38.py \
  --reference input/test36-dataset \
  --checkpoint results/train/checkpoint.pt \
  --generated results/generated-intermediate \
  --cgmd results/cgmd/md.extxyz \
  --output results/intermediate-analysis.json
```

`test38.py`にはtest37のようなCGMD生成サブコマンドはありません。このスクリプトの
`--cgmd`は既存のASE/extxyz軌道を解析するためのオプションです。

GitHubから既存のチェックアウトを更新する場合は、スパコン上で次を実行します。

```bash
cd /path/to/test38
git pull --ff-only origin main
```

途中学習の確認は、以下の1ジョブでチェックポイントのコピー・生成・解析まで実行します。
学習を止める必要はありません。`run_test38.pbs` の変更も不要です。

```bash
qsub -P <ProjectGroup_ID> test38_analysis.pbs
```

既定では `results/train/checkpoint.pt` を読み、なければ `train/checkpoint.pt` を探します。
別の保存先なら `-v TRAIN_DIR=/path/to/train` または `CHECKPOINT_PATH` を指定します。
ジョブ開始時の保存済みチェックポイントをコピーして固定するため、学習が進んでも
生成と解析は同じ重みを使用します。まだチェックポイントが保存されていない場合は終了します。

出力は毎回新しい `results/intermediate/run.XXXXXXXX/` に作成されます。
`checkpoint.pt` がコピーした重み、`generated/` が生成結果、`analysis.json` が解析結果です。
実際のパスはPBSログに表示します。GPU 1台・20時間を要求し、生成に18時間を割り当てます。
生成の時間予算に達した場合は保存済みの有効フレームだけを解析し、終了コード75を返します。
学習ジョブとは別のGPU割り当てを待つため、空き状況によっては待機します。

既存のCGMD軌道も同時に調べる場合は、`CGMD_PATH`を指定します。

```bash
qsub -P <ProjectGroup_ID> \
  -v CGMD_PATH=/scratch/my-run/md.extxyz \
  test38_analysis.pbs
```

## 範囲・出典

これは変位デノイザーによる構造生成であり、エネルギー・力場・物理的な時間を持つ
MDや平衡分布の検証を提供するものではありません。同梱データも平衡性を保証しません。
生成パラメーターは `test38.py` の実装・既定値を使用しています。

モデル等に含まれるDM2由来コードの出典は `test38.py` に記載し、
ライセンスを [licenses/DM2-LICENSE](licenses/DM2-LICENSE) に同梱しています。
GPU・PBS・コンテナ実行は実際のスパコンでの確認が必要です。

ローカルではPyTorch 2.6.0 / e3nn 0.4.4 / PyG 2.8.0.post1のCPU環境で、
4原子の合成データによる1更新の学習、2更新目への再開、2ステップ生成と
生成済みチェックポイントの再読み込みを確認しました。同梱データの形状と
ハッシュ、PBSスクリプトの構文・引数・入力不足時の終了も確認しています。
この確認は本番学習の収束や生成構造の妥当性を示すものではありません。
