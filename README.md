# Facial Expression Recognition Challenge

## დავალების აღწერა

ეს პროექტი მოიცავს FER-2013 (Facial Expression Recognition 2013) მონაცემთა ბაზაზე სახის ემოციების კლასიფიკაციას. მიზანი იყო ნულიდან ნეირონული ქსელის ოთხი სხვადასხვა არქიტექტურის დიზაინი, მომზადება და შეფასება — ყოველი მომდევნო მოდელი წინამდებარის სისუსტეებზე რეაგირებს და მათ გამოსწორებას ცდილობს. ყველა ექსპერიმენტი Weights & Biases-ის (W&B) მეშვეობით არის აღწერილი.

---

## პროექტის სტრუქტურა

```
model_experiment_simple_cnn.ipynb           --> საბაზისო CNN: baseline, ოპტიმიზატორის/LR-ის ცდა
model_experiment_batchnorm_dropout_cnn.ipynb --> BatchNorm + Dropout-ის დამატება overfitting-ის მოსაშორებლად
model_experiment_deep_cnn.ipynb             --> ღრმა VGG-ტიპის CNN + data augmentation + LR Scheduler
model_experiment_cnn_residual.ipynb         --> Residual კავშირებიანი ResNet-ტიპის CNN
model_inference.ipynb                       --> inference და submission
```

---

## მონაცემთა ბაზა

**FER-2013** შეიცავს 48×48 პიქსელიანი grayscale სახის სურათებს, რომელიც 7 ემოციად არის კლასიფიცირებული:

| კლასი | ემოცია   |
|-------|----------|
| 0     | Angry    |
| 1     | Disgust  |
| 2     | Fear     |
| 3     | Happy    |
| 4     | Sad      |
| 5     | Surprise |
| 6     | Neutral  |

დატასეტი შეიცავს **28,709** სურათს, საიდანაც 80/20 გაყოფით მივიღე:
- Train: **22,967** სურათი
- Validation: **5,742** სურათი

---

## წინდამუშავება

სურათები CSV ფაილში pixel-ების სტრიქონებად არის შენახული. preprocessing pipeline:

1. **Pixel Parsing** — თითოეული `pixels` სვეტის სტრიქონი გარდაიქმნება `48×48` numpy მასივად
2. **Normalization** — pixel მნიშვნელობები `[0, 255]`-დან `[0.0, 1.0]`-ამდე ნორმალიზდება (`/ 255.0`)
3. **Reshape** — `(N, 48, 48)` → `(N, 1, 48, 48)` PyTorch-ის `Conv2d` ფორმატისთვის
4. **Train/Val გაყოფა** — `train_test_split(test_size=0.2, random_state=42)`
5. **Dataset კლასი** — custom `FERDataset(Dataset)` კლასი `DataLoader`-ისთვის

---

## ექსპერიმენტები

### ექსპერიმენტი 1 — SimpleCNN (Baseline)
**notebook:** `model_experiment_simple_cnn.ipynb`

**მიზანი:** საბაზისო წერტილის დადგენა, რომ გავარკვიოთ „სად ვართ" ყოველგვარი რეგულარიზაციის გარეშე.

**არქიტექტურა:**
```
Conv2d(1→32, 3×3) → ReLU → MaxPool(2×2)
Conv2d(32→64, 3×3) → ReLU → MaxPool(2×2)
Flatten → Linear(9216→256) → ReLU → Linear(256→7)
```
- BatchNorm: არა
- Dropout: არა

**ჩატარებული ექსპერიმენტები (30 epoch, batch_size=64):**

| run სახელი | optimizer | learning_rate | batch_size |
|---|---|---|---|
| simple-cnn-adam-lr0.001 | Adam | 0.001 | 64 |
| simple-cnn-adam-lr0.0001 | Adam | 0.0001 | 64 |
| simple-cnn-sgd-lr0.01 | SGD (momentum=0.9) | 0.01 | 64 |
| simple-cnn-rmsprop-lr0.001 | RMSprop | 0.001 | 64 |
| simple-cnn-adam-batch128 | Adam | 0.001 | 128 |

**დაკვირვება:**
მოდელი კატასტროფულ overfitting-ს ავლენს — train accuracy ~99%, validation accuracy ~49%. ამ ფენომენის მიზეზი ნათელია: 2-ფენიანი CNN FER-2013-ის სირთულისთვის ძალიან მარტივია, ხოლო რეგულარიზაციის არარსებობა მოდელს "დაზეპირების" საშუალებას აძლევს. ვალიდაციის loss ეპოქებთან ერთად მუდმივად იზრდება, train loss კი ეცემა — overfitting-ის კლასიკური სიმპტომი. ეს მკაფიოდ მიუთითებს შემდეგ ნაბიჯზე: regularization.

---

### ექსპერიმენტი 2 — BNormDropoutCNN
**notebook:** `model_experiment_batchnorm_dropout_cnn.ipynb`

**მიზანი:** BatchNorm-ისა და Dropout-ის ეფექტის შესწავლა overfitting-ის შეკავებაში.

**არქიტექტურა:**
```
Conv Block 1: Conv(1→32) → BN → ReLU → Conv(32→32) → BN → ReLU → MaxPool → Dropout2d(p)
Conv Block 2: Conv(32→64) → BN → ReLU → Conv(64→64) → BN → ReLU → MaxPool → Dropout2d(p)
Conv Block 3: Conv(64→128) → BN → ReLU → MaxPool → Dropout2d(p)
FC: Flatten → Linear(4608→512) → ReLU → Dropout(2p) → Linear(512→128) → ReLU → Dropout(p) → Linear(128→7)
```

**ჩატარებული ექსპერიმენტები (30 epoch):**

| run სახელი | optimizer | LR | batch | dropout_rate |
|---|---|---|---|---|
| bndropout-adam-lr0.001-drop0.25 | Adam | 0.001 | 64 | 0.25 |
| bndropout-adam-lr0.001-drop0.5 | Adam | 0.001 | 64 | 0.50 |
| bndropout-adam-lr0.0001-drop0.25 | Adam | 0.0001 | 64 | 0.25 |
| bndropout-sgd-lr0.01-drop0.25 | SGD | 0.01 | 64 | 0.25 |
| bndropout-adam-batch128-drop0.25 | Adam | 0.001 | 128 | 0.25 |

**BatchNorm-ის როლი:** ნორმალიზაციით training სტაბილიზდება, gradient flow უმჯობესდება და მოდელი უფრო სწრაფად მიდის ოპტიმალური loss-სკენ (converge). BatchNorm ასევე მსუბუქი regularizer-ის ეფექტსაც ახდენს.

**Dropout-ის როლი:** შემთხვევითი ნეირონების „გათიშვა" training-ის დროს ხელს უშლის co-adaptation-ს — სიტუაციას, სადაც ნეირონები ერთმანეთის სპეციფიკურ სიგნალებს ზედმეტად „ეყრდნობიან", რაც overfitting-ის ერთ-ერთი მთავარი მიზეზია.

**დაკვირვება:**
regularization-ის დამატება საგრძნობლად ამცირებს train/val სხვაობას. dropout_rate=0.5 ზოგჯერ ზედმეტ regularization-ს იწვევს (underfitting). SGD momentum-ით Adam-ზე ნელ მილევას (converge) ვიღებთ.

---

### ექსპერიმენტი 3 — DeepCNN (VGG-ტიპის)
**notebook:** `model_experiment_deep_cnn.ipynb`

**მიზანი:** ქსელის გაღრმავება და data augmentation-ისა და LR Scheduler-ის შეტანა.

**არქიტექტურა:**
```
Block 1: Conv(1→64)×2    + BN + ReLU + MaxPool + Dropout
Block 2: Conv(64→128)×2  + BN + ReLU + MaxPool + Dropout
Block 3: Conv(128→256)×3 + BN + ReLU + MaxPool + Dropout  ← 3 conv ფენა (VGG-style)
Block 4: Conv(256→512)×2 + BN + ReLU + MaxPool + Dropout
FC: 4608 → 1024 → 512 → 128 → 7
```

**Data Augmentation:**
```python
RandomHorizontalFlip()
RandomRotation(degrees=10)
RandomAffine(degrees=0, translate=(0.1, 0.1))
```
augmentation მხოლოდ train-ზე გამოიყენება; val დატასეტი ხელუხლებელი რჩება.

**LR Scheduler:**
```
ReduceLROnPlateau(mode='min', factor=0.5, patience=5)
```
val loss-ის "გაჭედვის" შემთხვევაში LR-ი ავტომატურად ნახევრდება.

**ჩატარებული ექსპერიმენტები (50 epoch):**

| run სახელი | optimizer | LR | batch | dropout | augment |
|---|---|---|---|---|---|
| deep-cnn-adam-lr0.001-drop0.25 | Adam | 0.001 | 64 | 0.25 | კი |
| deep-cnn-adam-lr0.0001-drop0.25 | Adam | 0.0001 | 64 | 0.25 | კი |
| deep-cnn-adam-lr0.001-drop0.5 | Adam | 0.001 | 64 | 0.50 | კი |
| deep-cnn-adam-batch32-drop0.25 | Adam | 0.001 | 32 | 0.25 | კი |
| deep-cnn-adam-noaug-drop0.25 | Adam | 0.001 | 64 | 0.25 | არა |

**Data Augmentation-ის დამატების მოტივაცია:**
FER-2013-ში ადამიანები ყოველთვის არ იყურებიან ერთი კუთხით — ოდნავ შებრუნებული ან გადაადგილებული სახე ისევ იგივე ემოციის მატარებელია. augmentation დატასეტს ამ ვარიაციებს ხელოვნურად უმატებს, რაც მოდელს უფრო მნიშვნელოვან feature-ებს ასწავლის.

**`noaug` run-ის შედარება:** augmentation-ის გარეშე run უფრო სწრაფ overfitting-ს ავლენს. ეს ადასტურებს, რომ augmentation DeepCNN-ის სიღრმისთვის კრიტიკულია.

---

### ექსპერიმენტი 4 — ResNetCNN (Residual კავშირებით)
**notebook:** `model_experiment_cnn_residual.ipynb`

**მიზანი:** ResNet-ტიპის skip connection-ების გამოყენება gradient-ის უკეთ გავრცელებისა და ღრმა ქსელების ეფექტური სწავლებისთვის.

**ResidualBlock:**
```python
out = Conv → BN → ReLU → Conv → BN
residual = shortcut(x)
out = relu(out + residual)
out = Dropout2d(out)
```

**არქიტექტურა:**
```
initial: Conv(1→64) → BN → ReLU
Block 1: ResBlock(64→64)   + MaxPool
Block 2: ResBlock(64→128)  + MaxPool
Block 3: ResBlock(128→256) + ResBlock(256→256) + MaxPool  ← ორი residual block
Block 4: ResBlock(256→512) + MaxPool
FC: 4608 → 1024 → 512 → 128 → 7
```

channel-ის შეცვლის დროს (მაგ. 64→128) shortcut-ად 1×1 Conv გამოიყენება განზომილებების გასასწორებლად; ერთი და იგივე channel-ის შემთხვევაში — `nn.Identity()`.

**Data Augmentation და LR Scheduler** — იგივე, რაც DeepCNN-ში.

**ჩატარებული ექსპერიმენტები (50 epoch):**

| run სახელი | optimizer | LR | batch | dropout | augment |
|---|---|---|---|---|---|
| resnet-cnn-adam-lr0.001-drop0.25 | Adam | 0.001 | 64 | 0.25 | კი |
| resnet-cnn-adam-lr0.0001-drop0.25 | Adam | 0.0001 | 64 | 0.25 | კი |
| resnet-cnn-adam-lr0.001-drop0.5 | Adam | 0.001 | 64 | 0.50 | კი |
| resnet-cnn-adam-batch32-drop0.25 | Adam | 0.001 | 32 | 0.25 | კი |
| resnet-cnn-adam-noaug-drop0.25 | Adam | 0.001 | 64 | 0.25 | არა |

**Skip Connection-ების მოტივაცია:**
ღრმა ქსელებში gradient-ი backpropagation-ის დროს ადრეულ ფენებამდე ძლივს აღწევს (vanishing gradient). skip connection-ები gradient-ს პირდაპირ „ხიდს" უშენებს ადრეულ ფენებთან, რაც სწავლებას ასტაბილურებს. ამასთანავე, residual block-ს შეუძლია identity-ს სწავლა (output = input), თუ ფენა არაფერს უმატებს — ეს გაღრმავებას ნაკლებად „რისკიანს" ხდის.

---

## W&B Tracking

ყველა ექსპერიმენტი დალოგილია Weights & Biases პლატფორმაზე `Facial_Expression_Recognition` პროექტში. ლოგირებული მეტრიკები:

- `train/loss`, `train/accuracy` — ეპოქური სასწავლო შედეგები
- `val/loss`, `val/accuracy` — ეპოქური სავალიდაციო შედეგები
- `learning_rate` — LR Scheduler-ის დინამიკა (DeepCNN / ResNetCNN)
- `training_curves` — loss/accuracy გრაფიკი სურათის სახით
- `overfit_gap` — `train_acc - val_acc` (overfitting-ის რაოდენობრივი მაჩვენებელი)
- `best_val_accuracy`, `best_val_loss` — W&B Summary-ში

ყოველ run-ს აქვს `group` (მაგ. `simple-cnn`, `deep-cnn`) run-ების შედარების გასამარტივებლად.

WandB link: https://wandb.ai/ikvas22-free-university-of-tbilisi/Facial_Expression_Recognition?nw=nwuserikvas22

---

## არქიტექტურული პროგრესია

ყოველი ექსპერიმენტი წინამდებარის სისუსტეს პასუხობს:

| ეტაპი | მოდელი | ძირითადი სიახლე | პრობლემა, რომელსაც წყვეტს |
|---|---|---|---|
| 1 | SimpleCNN | Baseline | — |
| 2 | BNormDropoutCNN | BatchNorm + Dropout | კატასტროფული overfitting |
| 3 | DeepCNN | სიღრმე + Augmentation + LR Scheduler | განზოგადების სისუსტე |
| 4 | ResNetCNN | Skip Connections | Gradient Vanishing ღრმა ქსელებში |

---

## ტექნიკური გარემო

- **Framework:** PyTorch
- **გამოთვლა:** GPU (CUDA)
- **Tracking:** Weights & Biases (wandb)
- **მონაცემები:** Kaggle FER-2013 Competition (`train.csv`, `test.csv`)
- **Data Split:** `sklearn.model_selection.train_test_split` (80/20, random_state=42)
- **Loss Function:** `CrossEntropyLoss` (7-კლასიანი კლასიფიკაცია)
- **Optimizer:** Adam (baseline), SGD/RMSprop (comparative runs)
