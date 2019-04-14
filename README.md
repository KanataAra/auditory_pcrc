# auditory_rc
音響情報処理に特化したプレディクティブコーディングモデルを扱います。
hpcrcシリーズは2層の動的なプレディクティブコーディングモデルのコードです。hpcrcXXのXXの部分は、モデル番号を示します。

***

## 予測誤差を高次のネットワークで処理する構造

### hpcrc01.py
低次の層と高次の層を独立させて学習させる方式

### hpcrc02.py
低次の層と高次の層を結合させて学習させる方式

### hpcrc03.py
hpcrc02に高次の層のダイナミカルリザーバーのニューロンの自己結合の強度を定めるパラメータganma_rを追加

### hpcrc04.py & hpcrc05.py
低次の層と高次の層の結合にNMFを用いた方式

### hpcrc09.py
NMF+STPを組み込んだモデル

### hpcrc10.py
完成版（修論で用いたプログラム）

## 低次の出力を高次のネットワークで処理する構造

### hpcrc06.py
基本的な階層型モデル

### hpcrc07.py
NMFを組み込んだモデル

### hpcrc08.py
NMF+STPを組み込んだモデル