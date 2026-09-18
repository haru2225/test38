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
