# test39: 周期セル内の完全ノイズから生成する実験

`test39.py`はtest38の時間条件付きNequIP・CG種別・データ形式を再利用します。
test38で学習した0.001～0.75 Åの局所ノイズだけでは、セル全体に広がった
ランダム配置からの生成は学習範囲外でした。test39は周期セル上のブラウン運動で
ノイズを加え、ノイズが十分大きいときの終端分布をセル内の一様分布にします。
学習時の教師信号は周期境界を考慮した条件付きスコアです。生成では同じセル内の
一様ランダム配置から逆SDEを積分します。

入力はtest37と同じ `input/test36-dataset` 形式です。現在の実装は固定された
直交セルに限定しています。手元の粘土セルは20.64 × 35.864 × 31.0 Åで、
既定の最大ノイズは最長辺の35.864 Åです。そのとき最も遅く消える
フーリエモードの残差は約2.7×10^-9です。短い最大ノイズを指定して終端分布が
一様にならない設定は拒否します。

スパコンでは `test39.py`、`run_test39.pbs`、`test38.py` を同じ作業ディレクトリに
置き、test38と同じ `test38.sif` を利用できます。GPU 1台、20時間、
バッチサイズ2で学習します。標準では学習前に、粘土のO（ob, obos, oh, ohs）と
OHのH（ho）を除いたデータセットを作成します。

## Si・Alの周囲の酸素を省く粗視化

`qsub run_test39.pbs` の標準設定 `CG_MODE=cations` は、Si・Al・Mg・Na・Caの
中心位置を残します。現在のデータでは1292粒子から396粒子になります。
元のデータを維持し、`input/test39-cations/` に新しいデータを保存します。
原子種ID・対応表・ハッシュを更新し、すべてのフレームで同じ粒子を選択します。
2回目以降は元データと設定とハッシュが一致するデータを再利用します。

これは中心原子の座標を残す粗視化です。酸素を含むSiO4/AlO6の重心への置換や、
共有酸素の質量配分は行いません。出力の質量は残した原子の質量であり、CG分子動力学用に
較正した有効質量ではありません。省いた酸素の位置はこのモデルから復元できません。

```bash
qsub -P <課題番号> run_test39.pbs
# 中断した学習を同じ設定で再開
qsub -P <課題番号> -v RESUME=1 run_test39.pbs
# 学習後、別ジョブで完全ノイズから生成
qsub -P <課題番号> -v STAGE=generate run_test39.pbs
```

`results/test39-cations/train/checkpoint.pt` が学習結果です。生成出力は
`results/test39-cations/generated/` に入り、`generation.xyz` は一様ランダムな初期フレームを
含む全ステップの軌道です。`final.extxyz` は最終構造です。
時間予算・SIGTERMで停止した場合は有効フレームを保存し、終了コード75を返します。
再開時は設定と同じ出力ディレクトリを使用してください。

Oだけを除いてOHのHを残す場合は524粒子になります。この場合、Hは独立した残存サイトです。
学習・生成の両方に同じモードを指定してください。

```bash
qsub -v CG_MODE=oxygen-only run_test39.pbs
qsub -v CG_MODE=oxygen-only,STAGE=generate run_test39.pbs
```

酸素を残す従来の1292粒子モデルは `CG_MODE=none` で実行します。
その出力先は従来の `results/test39/` です。396粒子・524粒子モデルは
1292粒子のチェックポイントから再開できません。新しい出力先で再学習してください。

Pythonで前処理だけを実行する場合：

```bash
python test39.py coarse-grain --dataset input/test36-dataset --output input/test39-cations
python test39.py train --dataset input/test39-cations --output results/test39-cations/train --device cuda
```

Pythonの `train` は渡されたデータセットをそのまま使います。前処理の自動実行はPBS側で行います。

test38のチェックポイントは学習目標が違うためtest39の生成には使えません。
学習・生成プログラムが動くことと、粘土の結合・組成・層構造・RDFが正しいことは
別の検証です。現在のネットワークは10 Åの局所グラフなので、長距離秩序を
再現できる保証もありません。検証用フレームと生成構造を比較してください。

## 学習品質の改善オプション(すべてオプトイン、既定は変更なし)

学習が進んでも生成が収束しない(ランダムな配置のまま変化しない)場合の
診断で見つかった問題への対策を、**すべて既定オフの追加フラグ**として用意しています。
既に走っているジョブは何も指定しなければ`--resume`にそのまま使えます。

- **`--num-neighbors auto`**: `architecture()`内の近傍数の正規化定数は既定で
  `12`固定ですが、これは酸素を含む1292粒子系(test38)向けの値です。396粒子の
  カチオンのみモデルでは実際の10Å以内の平均近傍数は約73と大きく異なり、
  メッセージパッシングのスケールがずれたまま学習することになります。
  `auto`を指定するとデータセットの最初のフレームから実際の平均近傍数を
  計算して使います。数値を直接指定することもできます。
- **`--sigma-sampling log-normal`**: 生成を完全ランダムから始めるにはsigma_max
  (セル最長辺)が本当に必要ですが、既定の対数一様サンプリングだと
  sigma_minからsigma_maxまでの3桁近い範囲に学習の勾配更新が均等に散らばり、
  実際の粒子間距離(≈3Å)付近という「構造形成に一番効くが一番難しい」
  帯域への配分が薄くなります。`log-normal`にすると、sigma_maxはそのまま
  維持しつつ(終端分布が一様になる保証は変わらない)、学習時にサンプルする
  sigmaの分布だけをデータの典型スケール付近に集中させます。中心は既定で
  データセットの最近接距離の中央値を自動推定しますが、`--sigma-log-mean`/
  `--sigma-log-std`で手動指定もできます。
- **`--validation-frames`(既定8)**: `training.json`の`validation_mse`
  (small/middle/terminal)は元々検証フレーム1個だけの評価値でノイズが
  大きかったため、既定で複数フレームの平均に変更しました。こちらは
  チェックポイントの`settings`に影響しないため、`--resume`の互換性には
  影響しません。
- **`--irreps-hidden`/`--num-convs`**: 既定の`64x0e + 32x1e`・3層という
  構成は、test38の「参照構造にわずかなノイズを加えたところから戻す」
  局所デノイザー用に作られたものです。test39が要求する「セル内で完全に
  ランダムな配置から周期的な結晶格子を再構築する」というタスクは、
  はるかに難しい生成問題であり、この規模のネットワークでは表現力が
  足りていない可能性が高いです。`--irreps-hidden "128x0e + 64x1e + 32x2e"`
  のように広く、`--num-convs 5`のように深くすることで、まずネットワーク
  容量を増やしてみるのが最初の低リスクな対策です(NequIPには明示的な
  3体項(結合角の情報)がなく、角度的な相関は層を重ねるほど間接的に
  獲得されるため、層を増やすこと自体にも意味があります)。

これらは新規学習でのみ有効にしてください。既存のチェックポイントを
`--num-neighbors`・`--sigma-sampling`・`--irreps-hidden`・`--num-convs`を
変えて再開しようとすると、設定が一致しないため明示的にエラーになります
(黙って壊れた状態にはなりません)。

```bash
qsub -P <ProjectGroup_ID> \
  -v CG_MODE=cations,NUM_NEIGHBORS=auto,SIGMA_SAMPLING=log-normal,IRREPS_HIDDEN="128x0e + 64x1e + 32x2e",NUM_CONVS=5 \
  run_test39.pbs
```

## 学習途中のチェックポイントから、ランダム配置から生成される過程を確認する

`run_test39.pbs`での学習を止めずに、その時点で保存済みの`checkpoint.pt`だけを
使って生成できます。`test39_analysis.pbs`は現在の`checkpoint.pt`を別ディレクトリ
へコピーしてから(atomicなrenameで保存されているため、学習が同時に書き換えて
いても安全にコピーできます)、そのスナップショットで`test39.py generate`を
実行します。test38と共通の`test38.sif`をそのまま使います。

```bash
git pull --ff-only origin main
qsub -P <ProjectGroup_ID> test39_analysis.pbs
```

test39の生成は元々**必ず周期セル内の一様ランダムな配置から**始まります
(test38と違い、参照構造やノイズ幅を指定する`--initial-state`のような
オプションはありません。これがtest39の設計そのものです)。生成が完了または
時間切れで中断すると、`test39.py`自身が`generation.xyz`という拡張XYZ形式の
軌道ファイルを書き出します。1フレーム目(`generation_step=0`)がランダムな
初期配置、最終フレームが`final.extxyz`と同じ生成結果です。OVITOやVMDでこの
ファイルを開いて再生すると、ランダムな配置から粘土構造らしきものへ収束して
いく過程(またはまだ収束していない途中経過)を確認できます。ステップは生成
計算の番号であり、MDの物理時間ではありません。

既定では`results/test39-cations/train/checkpoint.pt`(`CG_MODE=cations`)を
読みます。`CG_MODE=oxygen-only`や`CG_MODE=none`で学習した場合は同じ変数を
指定してください。

```bash
qsub -P <ProjectGroup_ID> -v CG_MODE=oxygen-only test39_analysis.pbs
```

出力は毎回新しい`results/test39-cations/intermediate/run.XXXXXXXX/`に作成
されます。`checkpoint.pt`がコピーした重み、`generated/`が生成結果です。
実際のパスはジョブログに表示します。ジョブ開始時点の保存済みチェックポイント
を固定するため、その後も学習が進んでいてもこの生成結果は変わりません。
まだチェックポイントが保存されていない場合はエラーで終了します。

生成ステップ数(既定300)や時間予算は変数で変更できます。

```bash
qsub -P <ProjectGroup_ID> -v REVERSE_STEPS=100,GENERATE_TIME_HOURS=2 test39_analysis.pbs
```

学習ジョブとは別のGPU割り当てを待つため、空き状況によっては待機します。
