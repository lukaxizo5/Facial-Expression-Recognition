# Facial-Expression-Recognition

## პროექტის მოკლე აღწერა
ეს პროექტი არის supervised learning-ის ერთ-ერთი მაგალითი, რომელშიც ჩვენი მიზანია ადამიანის სახის გამოსახულებიდან ემოციის დადგენა.

ამოცანა წარმოადგენს multi-class image classification პრობლემას.
თითოეული სურათი არის 48x48 ზომის grayscale face image, ხოლო label შემდეგი ემოციებიდან ერთ-ერთი:

Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

### შეფასების მეტრიკა

მოდელი ფასდება Accuracy მეტრიკით.

Accuracy გვიჩვენებს მხოლოდ იმას, მონაცემების რა ნაწილი დავაკლასიფიცირეთ სწორად.

ამიტომ დამატებით დავაკვირდები:
* Macro F1 - რადგან კლასები დაბალანსებული არაა. Accuracy შეიძლება კარგი იყოს, მაგრამ იშვიათ კლასზე შედარებით ცუდად მუშაობდეს.
* Train/Validation Loss - ვნახოთ მოდელი სწავლობს თუ არა.
* Train/Validation Accuracy - მათ შორის სხვაობა გვეხმარება underfit/overfit გარჩევაში.
* Confusion Matrix - რათა ვნახოთ, რომელი ემოციები ერევა მოდელს ერთმანეთში.

შესაბამისად, მთავარი მეტრიკაა Accuracy, თუმცა მოდელის ანალიზისთვის გამოვიყენებ Macro F1 და Confusion Matrix.

## ჩემი მიდგომა
პრობლემის გადასაჭრელად გამოვიყენე შემდეგი workflow:
1. **EDA** - მონაცემების შესწავლა და პრობლემების იდენტიფიცირება
2. **Preprocessing** - Pixel string-ების 48x48 image tensor-ებად გადაყვანა და ნორმალიზება
3. **Baseline Model** - ძალიან მარტივი MLP მოდელის დატრენინგება
4. **CNN Models** - გადასვლა კონვოლუციურ არქიტექტურაზე
5. **Regularization and Augmentation** - Dropout, BatchNorm და light augmentation-ის დატესტვა
6. **Transfer Learning** - ResNet18-ის გამოყენება pretrained baseline მოდელად.

## რეპოზიტორიის სტრუქტურა

```
Facial-Expression-Recognition/
├── images/                          # გრაფიკები README-სთვის
├── data_exploration.ipynb           # Exploratory Data Analysis
├── mlp.ipynb                        # MLP baseline
├── tiny_cnn.ipynb                   # პირველი მარტივი CNN baseline
├── medium_cnn.ipynb                 # საშუალო ზომის CNN
├── deep_cnn.ipynb                   # დიდი ზომის CNN
├── resnet18_transfer.ipynb          # ResNet18 transfer learning ექსპერიმენტი
├── results_analysis.ipynb           # ყველა მოდელის შედეგების შედარება და ანალიზი
├── src/
│   ├── constants.py                 # პროექტში გამოყენებული constants
│   ├── dataset.py                   # FER2013 Dataset class და preprocessing
│   ├── metrics.py                   # accuracy, macro F1 და confusion matrix ფუნქციები
│   └── train_utils.py               # seed, device, train/evaluate loop-ები და parameter count
└── README.md                        # პროექტის დოკუმენტაცია
```


## Exploratory Data Analysis (EDA)

### მონაცემების სტრუქტურა
FER2013 dataset-ს აქვს 35887 სტრიქონი და **3 სვეტი**:
- emotion - target label, გვიჩვენებს რომელი ემოციაა სურათზე
- pixels - 48x48 grayscale სურათის pixel მნიშვნელობები string ფორმატში
- Usage - წინასწარ განსაზღვრული split: Training, PublicTest ან PrivateTest

Dataset წინასწარაა დაყოფილი Usage სვეტის მიხედვით, ამიტომ დამატებითი split აღარაა საჭირო.

| Split | Rows |
|---|---:|
| Training | 28709 |
| PublicTest | 3589 |
| PrivateTest | 3589 |

### Class Distribution
Class distribution შევამოწმე მხოლოდ Training data-ზე, რათა წინასწარ არ მენახა Validation/Test data.

Training data სრულად დაბალანსებული არაა. ყველაზე მეტი მაგალითი აქვს Happy კლასს, ხოლო ყველაზე ცოტა — Disgust კლასს. ეს ნიშნავს, რომ მხოლოდ Accuracy ყოველთვის საკმარისი არ არის მოდელის შესაფასებლად.
მოდელმა შეიძლება frequent class-ებზე კარგად იმუშაოს, მაგრამ იშვიათ კლასებზე ცუდი შედეგი ჰქონდეს.

ამიტომ Accuracy-სთან ერთად ვიყენებ Macro F1-საც. Macro F1 თითოეულ კლასს თანაბარ მნიშვნელობას ანიჭებს და უკეთ აჩვენებს, რამდენად კარგად მუშაობს მოდელი ყველა label-ზე და არა მხოლოდ ხშირ კლასებზე.

### Pixel Values and Normalization
პიქსელების მნიშვნელობები არის [0, 255] შუალედში. ნეირონული ქსელისთვის ასეთი დიდი scale არაა იდეალური და სჯობს preprocessing ეტაპზე პიქსელები დავასკალიროთ.
ჯერ პიქსელები გადავიყვანე [0, 1] შუალედში, ხოლო შემდეგ დავანორმალიზე.
ეს მონაცემები ჩავწერე constants.py ფაილში, რათა მომავალში გამოვიყენო.

### Sample Images
რამდენიმე ფოტო სამაგალითოდ ვნახე, რათა მცოდნოდა დაახლოებით როგორ მონაცემებთან მიწევდა მუშაობა.
დაბალი რეზოლუციის grayscale სურათები გვაქვს და ადამიანისთვისაც კი რთულია ზოგიერთი ემოციის გარჩევა, მაგალითად, Sad, Fear და Neutral.

შესაბამისად, დიდი ალბათობით მოდელსაც ყველაზე მეტად ეს კლასები აერევა ერთმანეთში. ამას შევამოწმებ Confusion Matrix-ით.


## MLP Underfit Baseline
ყველაზე მარტივ Baseline მოდელად გამოვიყენე ძალიან პატარა MLP.
ამ გაშვების მიზანი იყო underfit-ის ჩვენება.

მოდელი 48x48 ზომის სურათს უკეთებს flatten-ს, და შემდეგ 2 hidden unit-ში აკომპრესირებს.
არქიტექტურა:
```
Flatten -> Linear(2304 -> 2) -> ReLU -> Linear(2 -> 7)
```
მოდელს აქვს 4631 პარამეტრი.

შედეგი:
```
Hidden Size - 2
Parameters - 4631
Best Val. Accuracy - 0.3508
Best Val Macro F1 - 0.2153
```

ამ მოდელში train და validation accuracy ორივე დაბალია.
საბოლოო train accuracy არის 35.27%, ხოლო საუკეთესო validation accuracy - 35.08%.
მათ შორის პატარა სხვაობაა და ორივე დაბალია, რაც ნიშნავს, რომ მოდელი საკმარისად კარგად ვერ სწავლობს მონაცემებს შორის არსებულ კავშირს.

ეს არის underfit-ის კარგი მაგალითი. მოდელს აქვს მცირე capacity, 48x48 სურათი უნდა შეკუმშოს 2 hidden unit-ში და flatten-ის გამო ვერ ინარჩუნებს Spatial, ანუ სივრცულ სტრუქტურას.

Macro F1-იც დაბალია (0.2153), რაც ნიშნავს, რომ ყველა კლასზე თანაბრად კარგად არ მუშაობს.

შემდეგ ეტაპზე გადავალ CNN-ზე, რადგან ის შეეფერება სურათის კლასიფიკაციის ამოცანას spatial სტრუქტურების სწავლის უნარის გამო.


## Tiny CNN

TinyCNN არის პირველი კონვოლუციური მოდელი პროექტში. MLP-გან განსხვავებით, ის სურათს მაშინვე flatten-ს არ უკეთებს.
ჯერ კონვოლუციური layer-ებით სწავლობს ლოკალურ spatial patterns-ს და შემდეგ classification head პროგნოზირებს 7 კლასს.

არქიტექტურა:
```
Conv2d(1 -> 16) -> ReLU -> MaxPool
Conv2d(16 -> 32) -> ReLU -> MaxPool
Flatten -> Linear(4608 -> 128) -> ReLU -> Dropout -> Linear(128 -> 7)
```
მოდელს აქვს 595,655 პარამეტრი.

პირველ რიგში გავუშვი small-data sanity check. ლექციაზე ვთქვით, რომ საკმარისი capacity მქონე ნეირონულ ქსელს უნდა შეეძლოს პატარა დატასეტის დაზეპირება.
ამის შესამოწმებლად მოდელი დავატრენინგე 50 სურათზეც და შეფასებაც იმავე 50 სურათზე გავაკეთე.
მოდელმა მიაღწია 100% accuracy-ს, რაც ნიშნავს, რომ მას შეუძლია პატარა დატასეტის დაზეპირება.

| Run | Learning Rate | Dropout | Best Val Accuracy | Best Val Macro F1 |
|---|---:|---:|---:|---:|
| TinyCNN small-50 overfit sanity check | 1e-3 | 0.0 | - | - |
| TinyCNN baseline | 1e-3 | 0.0 | 0.52689 | 0.51353 |
| TinyCNN lr=3e-4 | 3e-4 | 0.0 | 0.52466 | 0.49788 |
| TinyCNN lr=3e-3 | 3e-3 | 0.0 | 0.50125 | 0.47180 |
| TinyCNN dropout=0.3 | 1e-3 | 0.3 | 0.53692 | 0.51132 |

საუკეთესო validation accuracy მიიღო TinyCNN dropout=0.3 მოდელმა — 0.53692.

MLP underfit baseline-თან შედარებით TinyCNN-მ დიდი გაუმჯობესება აჩვენა:

MLP best val accuracy:     0.3508
TinyCNN best val accuracy: 0.5369

ეს აჩვენებს, რომ კონვოლუციური layers ბევრად უკეთესია image classification ამოცანისთვის, რადგან ისინი სურათის spatial structure-ს იყენებენ.

საინტერესოა, რომ dropout-მა train accuracy შეამცირა, მაგრამ validation accuracy გააუმჯობესა. Baseline TinyCNN-ს train accuracy ჰქონდა დაახლოებით 90%, ხოლო validation accuracy მხოლოდ 51%, რაც overfitting-ზე მიუთითებს. Dropout=0.3 შემთხვევაში train accuracy შემცირდა დაახლოებით 72%-მდე, მაგრამ validation accuracy გაიზარდა 53.7%-მდე.
ეს ნიშნავს, რომ dropout-მა generalization გააუმჯობესა, რაც ლოგიკურია, რადგან ensemble-ს ჰგავს.



### MediumCNN
TinyCNN-ის შემდეგ ვცადე უფრო ძლიერ CNN არქიტექტურა.
MediumCNN-ს აქვს მეტი კონვოლუციური layer და channel, რაც მოდელს უფრო რთული ვიზუალური პატერნების სწავლის შესაძლებლობას აძლევს.

TinyCNN-ში გამოყენებული იყო ორი კონვოლუციური layer, ხოლო MediumCNN-ში გვაქვს სამი კონვოლუციური ბლოკი და თითოეულ ბლოკში ორი კონვოლუციური layer.

არქიტექტურა:
```
ConvBlock(1 -> 32)
ConvBlock(32 -> 64)
ConvBlock(64 -> 128)
Flatten
Linear(4608 -> 256)
ReLU
Dropout
Linear(256 -> 7)
```

თითოეული კონვოლუციური ბლოკი შედგება შემდეგი layer-ებისგან:

- Conv2d
- BatchNorm2d
- ReLU
- Conv2d
- BatchNorm2d
- ReLU
- MaxPool2d

მოდელს აქვს 1,469,031 პარამეტრი.

### ძირითადი გადაწყვეტილებები
- გავზარდე კონვოლუციური layer-ების რაოდენობა, რადგან საწყისები სწავლობენ უფრო მარტივ feature-ებს (კონტურები და ა.შ.), ხოლო შემდეგები უფრო რთულებს (პირის ფორმა და ა.შ.).
- დავამატე BatchNorm2d რათა ტრენინგი უფრო სტაბილური ყოფილიყო. ის აქტივაციებს ანორმალიზებს და მოდელს უფრო სწრაფად სწავლაში ეხმარება.
- გავტესტე სხვადასხვა რეგულარიზაციის მეთოდები: dropout, weight decay, horizontal flip, learning rate.
- ყველა run გავუშვი 15 ეპოქაზე.

### შედეგები და მათი ანალიზი

| Run | Learning Rate | Dropout | Weight Decay | Augmentation | Best Val Accuracy | Best Val Macro F1 |
|---|---:|---:|---:|---|---:|---:|
| MediumCNN baseline | 1e-3 | 0.0 | 0.0 | none | 0.61060 | 0.59854 |
| MediumCNN dropout=0.3 | 1e-3 | 0.3 | 0.0 | none | 0.61967 | 0.58124 |
| MediumCNN dropout=0.5 | 1e-3 | 0.5 | 0.0 | none | 0.61230 | 0.50553 |
| MediumCNN weight decay=1e-4 | 1e-3 | 0.3 | 1e-4 | none | 0.62430 | 0.56979 |
| MediumCNN horizontal flip | 1e-3 | 0.3 | 0.0 | hflip | 0.62440 | 0.58553 |
| MediumCNN learning rate=3e-4 | 3e-4 | 0.3 | 0.0 | none | 0.59920 | 0.58245 |

- Baseline მოდელმა validation accuracy მიხედვით მიიღო დაახლოებით 61.1%, ხოლო საუკეთესო Macro F1 იყო 0.59854.
თუმცა საბოლოო train accuracy დაახლოებით 94.9% იყო, validation accuracy კი დაახლოებით 61%. მათ შორის დიდი სხვაობა მიუთითებს overfitting-ზე.
ასევე საბოლოო validation loss გაიზარდა დაახლოებით 1.86-მდე, მიუხედავად იმისა, რომ training loss ძალიან დაბალი იყო. ეს ნიშნავს, რომ მოდელი training data-ს ძალიან კარგად სწავლობს, მაგრამ validation data-ზე განზოგადებას ვერ აკეთებს.
ზ
- Dropout = 0.3 დამატების შემდეგ train accuracy შემცირდა დაახლოებით 80.4%-მდე, მაგრამ validation accuracy გაიზარდა დაახლოებით 62%-მდე.
ეს მოსალოდნელი შედეგია, რადგან Dropout training-ს ართულებს და train metric-ს ამცირებს, თუმცა მოდელს ეხმარება განზოგადებაში.

- Dropout=0.5 dropout=0.3-ზე უარესი აღმოჩნდა. განსაკუთრებით შემცირდა Macro F1 და გახდა დაახლოებით 0.50553.
ეს ნიშნავს, რომ 0.5 Dropout ამ ზომის არქიტექტურისთვის ზედმეტად ძლიერი რეგულარიზაცია გამოვიდა. მოდელი training-ის დროს ზედმეტ ინფორმაციას კარგავს და ყველა კლასის feature-ებს საკმარისად კარგად ვეღარ სწავლობს.

- Weight decay დამატებამ ერთ-ერთი საუკეთესო validation accuracy მოგვცა — დაახლოებით 0.6243.
Train accuracy baseline-თან შედარებით მნიშვნელოვნად დაბალი იყო, რაც ნიშნავს, რომ weight decay-მა მოდელს training data-ს ზედმეტად დამახსოვრების საშუალება არ მისცა.

- Horizontal Flip Augmentation-მა საუკეთესო validation accuracy აჩვენა — დაახლოებით 0.6244.
ამ run-ში საბოლოო train accuracy დაახლოებით 68.7% იყო, validation accuracy კი დაახლოებით 62.1%. Train და validation შედეგებს შორის პატარა სხვაობა მიუთითებს, რომ augmentation-მა overfitting მნიშვნელოვნად შეამცირა.
Augmentation-ის დროს მოდელი ყოველ ეპოქაზე ოდნავ განსხვავებულ მონაცემებს ხედავს, ამიტომ კონკრეტული პიქსელების მიმდევრობის დამახსოვრების ნაცვლად უფრო ზოგადი facial features-ების სწავლა უწევს.

Accuracy მიხედვით, საუკეთესო MediumCNN run იყო run_11_medium_cnn_hflip:
- Best Validation Accuracy = 0.6244
- Best Validation Macro F1 = 0.58553

### TinyCNN-თან შედარება

TinyCNN-ის საუკეთესო validation accuracy იყო 0.5369

MediumCNN-ის საუკეთესო validation accuracy გახდა 0.6244

გვაქვს გაუმჯობესება 0.6244 - 0.5369 = 0.0875

ეს აჩვენებს, რომ უფრო ღრმა კონვოლუციური feature extractor და მეტი channel მოდელს ეხმარება facial expression-ების უკეთ ამოცნობაში.

თუმცა baseline run-ის შედეგებმა ასევე აჩვენა, რომ model capacity-ის გაზრდა overfitting-ის რისკსაც ზრდის. ამიტომ უფრო დიდი architecture-ისთვის Dropout, Weight Decay და Data Augmentation მნიშვნელოვანი ხდება.


## Deeper CNN
MediumCNN-ის შემდეგ კიდევ ერთი უფრო ღრმა CNN ვცადე.
DeepCNN-ში დამატებულია მეოთხე კონვოლუციური ბლოკი, რომლის მიზანიცაა უფრო მაღალი დონის feature-ბის სწავლა და შემოწმება, აუმჯობესებს თუ არა არქიტექტურის კომპლექსურობის გაზრდა შედეგს.

არქიტექტურა:
```
ConvBlock(1 -> 32)
ConvBlock(32 -> 64)
ConvBlock(64 -> 128)
ConvBlock(128 -> 256)
Flatten
Linear(2304 -> 256)
ReLU
Dropout
Linear(256 -> 7)
```

თითოეული კონვოლუციური ბლოკი შედგება შემდეგი layer-ებისგან:

- Conv2d
- BatchNorm2d
- ReLU
- Conv2d
- BatchNorm2d
- ReLU
- MaxPool2d

მოდელს აქვს 1,765,479 პარამეტრი.

### ძირითადი გადაწყვეტილებები
- დავამატე მეოთხე კონვოლუციური ბლოკი და channel-ების რაოდენობა 256-მდე გავზარდე.
- overfit-ის შესამცირებლად გავტესტე Dropout, Weight decay და light augmentation.
- Adam Optimizer შევადარე RMSProp-ს.
- ბოლო run-ში გამოვიყენე ReduceLROnPlateau, რაც validation loss გაუმჯობესების შეჩერებისას learning rate ამცირებს.
- ყველა run გავუშვი 15 ეპოქაზე ბოლოს გარდა, ბოლო 20-ზე.

### შედეგები და მათი ანალიზი

| Run | Learning Rate | Dropout | Weight Decay | Augmentation | Optimizer | Scheduler | Epochs | Val Accuracy | Val Macro F1 |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| DeepCNN baseline | 1e-3 | 0.0 | 0.0 | none | Adam | none | 15 | 0.60602 | 0.58517 |
| DeepCNN dropout=0.3 | 1e-3 | 0.3 | 0.0 | none | Adam | none | 15 | 0.60379 | 0.58008 |
| DeepCNN weight decay=1e-4 | 1e-3 | 0.3 | 1e-4 | none | Adam | none | 15 | 0.59682 | 0.57886 |
| DeepCNN light augmentation | 1e-3 | 0.3 | 1e-4 | light | Adam | none | 15 | 0.61410 | 0.55504 |
| DeepCNN RMSprop | 1e-3 | 0.3 | 1e-4 | light | RMSprop | none | 15 | 0.53970 | 0.44532 |
| DeepCNN Adam + scheduler | 1e-3 → 5e-4 | 0.3 | 1e-4 | light | Adam | ReduceLROnPlateau | 20 | 0.65422 | 0.61175 |

- Baseline DeepCNN-ის საბოლოო train accuracy იყო დაახლოებით 95.3%, ხოლო validation accuracy — დაახლოებით 60.6%.
Training loss შემცირდა 0.14265-მდე, მაგრამ validation loss გაიზარდა 1.98704-მდე.
Train და validation შედეგებს შორის დიდი სხვაობა აჩვენებს, რომ მოდელი overfit-ში წავიდა.
ანუ არქიტექტურის გაღრმავება საკმარისი არაა, დიდი capacity მოდელს training data-ს დამახსოვრებაში ეხმარება, მაგრამ ავტომატურად უფრო განზოგადებადს არ ხდის.

- Dropout-ის დამატების შემდეგ train accuracy შემცირდა დაახლოებით 88.7%-მდე, თუმცა validation accuracy მნიშვნელოვანი რაოდენობით არ გაუმჯობესებულა და გახდა 0.60379.
Validation loss baseline-თან შედარებით შემცირდა, მაგრამ მოდელი მაინც overfit-შია.
ანუ მხოლოდ dropout არ აღმოჩნდა საკმარისი რეგულარიზაცია validation accuracy-ს გასაუმჯობესებლად.

- Weight Decay-ის დამატების შემდეგ train accuracy დაახლოებით 86.3% გახდა, ხოლო validation accuracy — 0.59682.
Weight Decay-მ მოდელის წონები შეზღუდა და training accuracy შეამცირა, მაგრამ ამ კონკრეტულ კონფიგურაციაში validation accuracy არ გაუმჯობესებულა.
ეს არ ნიშნავს, რომ Weight Decay ყოველთვის საზიანოა, უბრალოდ მისი ეფექტი დამოკიდებულია სხვა ჰიპერპარამეტრებზეც.

- Light augmentation მოიცავდა:
  1. Horizontal Flip-ს
  2. მცირე rotation-ს
  3. მცირე translation-ს
  4. მცირე scale ცვლილებას.

ამ run-ში train accuracy შემცირდა დაახლოებით 63.9%-მდე, ხოლო validation accuracy გაიზარდა 0.61410-მდე.
Baseline მოდელთან შედარებით train და validation შედეგებს შორის სხვაობა მნიშვნელოვნად შემცირდა.
ეს ნიშნავს, რომ augmentation-მა overfitting შეამცირა.
თუმცა აღსანიშნავია, რომ ამან MediumCNN-ის საუკეთესო შედეგს მაინც ვერ გადააჭარბა.

- RMSProp-მა ყველაზე სუსტი შედეგი აჩვენა:
  - Validation Accuracy: 0.53970
  - Validation Macro F1: 0.44532

ანუ ამ კონკრეტულ კონფიგურაციაში RMSprop Adam-ზე მნიშვნელოვნად უარესი აღმოჩნდა.

- Adam + Learning Rate Scheduler-მა საუკეთესო შედეგი აჩვენა.
ამ კონფიგურაციაში გამოვიყენეთ
  - Adam optimizer;
  - Dropout = 0.3;
  - Weight Decay = 1e-4;
  - Light Augmentation;
  - ReduceLROnPlateau scheduler;
  - 20 epoch.

Scheduler-მა learning rate 1e-3-დან 5e-4-მდე შეამცირა.

საბოლოო შედეგები იყო:
- Train Accuracy: 0.69052
- Validation Accuracy: 0.65422
- Validation Macro F1: 0.61175
- Validation Loss: 0.96115

ამ შემთხვევაში train accuracy validation accuracy-თან შედარებით ძალიან მაღალი აღარ იყო და validation loss სხვა DeepCNN run-ებთან შედარებით ყველაზე დაბალი აღმოჩნდა.

ეს აჩვენებს, რომ DeepCNN-თვის დამატებითი epochs სასარგებლო გახდა მხოლოდ მაშინ, როდესაც learning rate ტრენინგის პროცესში შემცირდა.

### MediumCNN-თან შედარება

MediumCNN-ის საუკეთესო validation accuracy იყო 0.6244

DeepCNN-ის საუკეთესო validation accuracy გახდა 0.65422

გვაქვს გაუმჯობესება 0.65422 - 0.62440 = 0.02982

MediumCNN-ის საუკეთესო run-ის Macro F1 იყო 0.58553, ხოლო DeepCNN scheduler run-ის Macro F1 გახდა 0.61175.

გვაქვს გაუმჯობესება 0.61175 - 0.58553 = 0.02622.

ეს ნიშნავს, რომ DeepCNN scheduler configuration უკეთ მუშაობს არა მხოლოდ Accuracy-ის მიხედვით, არამედ კლასებს შორის უფრო დაბალანსებულ შედეგსაც აჩვენებს.

ამ ექსპერიმენტიდან ჩანს, რომ მოდელის კომპლექსურობა (სიღრმე) სასარგებლოა მხოლოდ მაშინ, როდესაც უფრო დიდი capacity შესაბამის რეგულარიზაციასა და learning-rate კონტროლთან ერთად გამოიყენება.

### Final DeepCNN Run

ბოლო კონფიგურაციის run გავუშვი უფრო მეტ (30) ეპოქაზე, რადგან მჯეროდა ამ კონფიგურაციას თავის საუკეთესო შედეგი ჯერ არ ჰქონდა დაფიქსირებული.
ასევე დავამატე Early Stopping, ანუ ტრენინგი შეწყდეს, თუ validation accuracy ზედიზედ 6 ეპოქის განმავლობაში არ გაუმჯობესდება.

საბოლოო run-ის შედეგები იყო:
- Train Accuracy: 0.75530
- Validation Accuracy: 0.67930
- Validation Macro F1: 0.65531
- Validation Loss: 0.94356

Train და Validation accuracy შორის სხვაობა გაიზარდა, რაც მოსალოდნელი იყო, თუმცა validation accuracy, macro F1 და validation loss გაუმჯობესდა.

შესაბამისად, ესაა საბოლოო და საუკეთესო DeepCNN კონფიგურაცია.


## ResNet18 Transfer Learning
CNN შემდეგ გადავედი transfer learning-ზე და გამოვიყენე ImageNet-ზე pretrained ResNet18.

FER2013-ის სურათები არის 1x48x48 grayscale ფორმატში, ხოლო pretrained ResNet18 მოელის 3x224x224 სურათს.
ამიტომ preprocessing-ის დროს:
- grayscale channel გავიმეორე სამჯერ
- სურათი გავზარდე 224x224 ზომამდე
- გამოვიყენე ImageNet ნორმალიზაცია

ResNet18-ის თავდაპირველი classifier შექმნილია ImageNet-ის 1000 კლასისთვის. ის შევცვალე FER2013-ის 7 emotion class-ზე მორგებული classifier-ით:
- Original classifier: Linear(512 -> 1000)
- New classifier: Dropout Linear(512 -> 7)

ResNet18-ს აქვს residual connections, რომლებიც ბლოკის input-ს output-ს პირდაპირ უმატებს. ეს გრადიენტსს ღრმა network-ში გადაადგილებას უმარტივებს და ღრმა არქიტექტურის ტრენინგს უფრო სტაბილურს ხდის.

მოდელს აქვს 11,180,103 პარამეტრი.

### ძირითადი გადაწყვეტილებები
გავტესტე pretrained ResNet18-ის გამოყენების სამი ძირითადი მიდგომა:
- Frozen backbone - pretrained network მთლიანად გაყინული რჩება და მხოლოდ ახალი კლასიფიკატორი სწავლობს.
- Layer4 fine-tuning - სწავლობს ბოლო residual ბლოკი და კლასიფიკატორი.
- Full fine-tuning - სწავლობს მთელი pretrained network.

ასევე, საბოლოო კონფიგურაციაში გამოვიყენე:
- Light Augmentation
- Dropout = 0.3
- Weight Decay = 1e-4
- განსხვავებული learning rate კლასიფიკატორისა და backbone-თვის
- ReduceLROnPlateau scheduler
- Early Stopping
- საუკეთესო validation accuracy-თვის checkpoint შენახვა

### შედეგები და მათი ანალიზი

### ResNet18 Final-Epoch Results

| Run | Fine-tuning Strategy | Total Parameters | Trainable Parameters | Train Accuracy | Train Loss | Train Macro F1 | Val Accuracy | Val Loss | Val Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet18 frozen backbone | frozen | 11,180,103 | 3,591 | 0.42018 | 1.52040 | 0.34761 | 0.44469 | 1.48305 | 0.36898 |
| ResNet18 layer4 fine-tuning | layer4 | 11,180,103 | 8,397,319 | 0.92821 | 0.21688 | 0.92672 | 0.64726 | 1.72627 | 0.63329 |
| ResNet18 full fine-tuning | full | 11,180,103 | 11,180,103 | 0.92354 | 0.22369 | 0.91844 | 0.65450 | 1.55305 | 0.63202 |
| ResNet18 full augmentation + scheduler | full | 11,180,103 | 11,180,103 | 0.86551 | 0.37221 | 0.86723 | 0.69156 | 1.11841 | 0.68460 |

- Frozen backbone configuration-ში სულ 3,591 პარამეტრი იყო, რადგან მხოლოდ ახალი კლასიფიკატორი სწავლობდა.
საუკეთესო validation accuracy იყო 0.46893, ხოლო Macro F1 — 0.37718.
ეს მნიშვნელოვნად დაბალი შედეგია სხვა ResNet18 run-ებთან შედარებით. ImageNet-ზე ნასწავლი ზოგადი feature-ები გარკვეულ ინფორმაციას შეიცავს, თუმცა მხოლოდ კლასიფიკატორის training საკმარისი არ აღმოჩნდა დაბალი რეზოლუციის grayscale facial expression-ებზე გადასატანად.

- Layer4-ისა  და კლასიფიკატორის fine-tuning-მა შედეგი მნიშვნელოვნად გააუმჯობესა:
  - საუკეთესო Validation Accuracy: 0.65729
  - საუკეთესო Validation Macro F1: 0.63071

  აქ 8,397,319 პარამეტრი იყო. ბოლო residual ბლოკის განბლოკვამ მოდელს მისცა საშუალება, ImageNet-ზე ნასწავლი მაღალი დონის feature-ები FER2013-ის facial expression-ებზე მოერგო.
  Early Stopping გამო training 12-ის ნაცვლად 10 ეპოქაში დასრულდა. საუკეთესო შედეგი დაფიქსირდა მე-5 ეპოქაზე.

- Full fine-tuning დროს მთელი 11,180,103 პარამეტრი სწავლობდა და მივიღე შედეგი:
  - საუკეთესო Validation Accuracy: 0.66007
  - საუკეთესო Validation Macro F1: 0.63005

  Validation accuracy მცირედით გაუმჯობესდა layer4 fine-tuning-თან შედარებით, თუმცა Macro F1 თითქმის იგივე დარჩა.
  ეს აჩვენებს, რომ მთელი network-ის ადაპტაცია სასარგებლო იყო, თუმცა augmentation და scheduler გარეშე გაუმჯობესება მცირე აღმოჩნდა.

- Full fine-tuning + Augmentation + Scheduler-ს საუკეთესო შედეგი ჰქონდა:
  გამოვიყენე:
  - სრული network-ის fine-tuning
  - horizontal flip
  - მცირე rotation
  - მცირე translation
  - მცირე scale ცვლილება
  - Dropout = 0.3
  - Weight Decay = 1e-4
  - Backbone Learning Rate = 3e-5
  - Head Learning Rate = 3e-4
  - ReduceLROnPlateau scheduler
  - 20 ეპოქა

  საუკეთესო შედეგი მივიღე მე-18 ეპოქაზე:
  - საუკეთესო Validation Accuracy: 0.69184
  - საუკეთესო Validation Macro F1: 0.68462

  ეს არის ყველა ჩატარებულ ექსპერიმენტს შორის საუკეთესო შედეგი.

  საბოლოო ეპოქაზე მივიღე:
  - Train Accuracy: 0.8655
  - Validation Accuracy: 0.6916
  - Train Loss: 0.3722
  - Validation Loss: 1.1184

  train და validation შორის ამხელა სხვაობა overfit-ზე მიუთითებს,
  Validation loss დაახლოებით მე-7 ეპოქის შემდეგ ზრდას იწყებს, მიუხედავად იმისა, რომ validation accuracy მცირე რაოდენობით კიდევ უმჯობესდება.
  ეს შესაძლებელია იმიტომ, რომ Cross-Entropy Loss ითვალისწინებს არა მხოლოდ prediction სისწორეს, არამედ confidence-საც.
  მოდელმა შეიძლება მეტი image სწორად ამოიცნოს, მაგრამ დარჩენილ შეცდომებზე უფრო მაღალი confidence ჰქონდეს. ასეთ შემთხვევაში accuracy იზრდება, ხოლო loss შეიძლება გაუარესდეს.

  რადგან ძირითადი მეტრიკაა accuracy, საბოლოო მოდელად ვიყენებ მე-18 ეპოქაზე შენახულ checkpoint-ს.


### არქიტექტურების საერთო შედარება

| Architecture | Best Val Accuracy | Best Val Macro F1 |
|---|---:|---:|
| MLP | 0.35080 | 0.21530 |
| TinyCNN | 0.53692 | 0.51132 |
| MediumCNN | 0.62440 | 0.58553 |
| DeepCNN | 0.67930 | 0.65531 |
| ResNet18 | 0.69184 | 0.68462 |

მოდელის სირთულის ზრდასთან ერთად validation შედეგიც ეტაპობრივად გაუმჯობესდა.
MLP სურათის სივრცულ სტრუქტურას არ იყენებდა და ყველაზე სუსტი შედეგი აჩვენა.
Custom CNN-ებმა კონვოლუციური feature extraction-ის გამოყენებით მნიშვნელოვანი გაუმჯობესება მოგვცა.
DeepCNN-მა scheduler-ისა და უფრო ხანგრძლივი training-ის გამოყენებით დაახლოებით 67.9% accuracy მიიღო.
საბოლოოდ pretrained ResNet18-მა საუკეთესო Accuracy და Macro F1 მოგვცა.

## საბოლოო მოდელი
საბოლოო model-ად ავირჩიე:
- ResNet18
- Full Fine-tuning
- Light Augmentation
- Dropout = 0.3
- Weight Decay = 1e-4
- ReduceLROnPlateau
- Best Checkpoint Epoch = 18

საბოლოო validation შედეგებია:
- Validation Accuracy = 69.18%
- Validation Macro F1 = 68.46%

ამ კონფიგურაციამ აჩვენა საუკეთესო validation accuracy და macro F1, ამიტომ ვიყენებ საბოლოო შეფასებისთვის.




## შედეგები

საბოლოო ResNet18 checkpoint შეფასდა `PrivateTest` split-ზე ერთხელ.

მიღებული შედეგები:

PrivateTest Loss = 0.99343
PrivateTest Accuracy = 0.71106
PrivateTest Macro F1 = 0.70811

Validation შედეგებთან შედარებით ორივე მეტრიკა გაიზარდა, რაც ნიშნავს, რომ მოდელმა დამოუკიდებელ test split-ზეც სტაბილური შედეგი აჩვენა.

## Per-class შედეგები

| Emotion | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Angry | 0.65086 | 0.61507 | 0.63246 | 491 |
| Disgust | 0.83333 | 0.72727 | 0.77670 | 55 |
| Fear | 0.59471 | 0.55303 | 0.57311 | 528 |
| Happy | 0.90183 | 0.89875 | 0.90029 | 879 |
| Sad | 0.53130 | 0.58586 | 0.55725 | 594 |
| Surprise | 0.83538 | 0.81731 | 0.82625 | 416 |
| Neutral | 0.67901 | 0.70288 | 0.69074 | 626 |

ყველაზე მაღალი F1-score მიიღო Happy კლასმა - დაახლოებით 0.9003.
ასევე ძლიერი შედეგი ჰქონდა Surprise კლასს - დაახლოებით 0.8262.

ყველაზე რთული კლასები აღმოჩნდა Sad და Fear, რომელთა F1-score შესაბამისად იყო 0.55725 და 0.57311.

Disgust კლასის შედეგი მაღალია, თუმცა ამ კლასს მხოლოდ 55 მონაცემი აქვს, ამიტომ მისი მეტრიკები შედარებით მცირე რაოდენობის მაგალითებზეა გამოთვლილი.


ქვემოთ მოცემულია PrivateTest split-ის normalized confusion matrix:

![Confusion Matrix](/images/final_confusion_matrix.png)

Confusion matrix-იდან ჩანს, რომ:

- Happy ყველაზე მარტივად ამოსაცნობი emotion იყო და შემთხვევების დაახლოებით 90% სწორად კლასიფიცირდა.
- Surprise შემთხვევების დაახლოებით 82% სწორად ამოიცნო მოდელმა.
- Fear ხშირად ერეოდა Sad კლასში.
- Sad ხშირად ერეოდა Neutral და Fear კლასებში.
- Angry შემთხვევების ნაწილი მოდელმა Sad-ად დააკლასიფიცირა.
- Neutral და Sad ერთმანეთთან შედარებით ხშირად ერეოდა.

ეს შედეგები მოსალოდნელია, რადგან დაბალი რეზოლუციის grayscale სურათებში Fear, Sad და Neutral ვიზუალურად ხშირად ერთმანეთის მსგავსია.


## W&B

ყველა experiment, training curve, validation metric, confusion matrix, per-class result და final model checkpoint დალოგილია Weights & Biases-ში.

W&B Project Link: https://wandb.ai/lkhiz23-free-university-of-tbilisi-/facial-expression-recognition/overview

საბოლოო evaluation run: run_26_resnet18_private_test

W&B-ში შენახულია:

- PrivateTest Accuracy
- PrivateTest Macro F1
- PrivateTest Loss
- Confusion Matrix
- თითოეული class-ის Precision, Recall და F1-score
- საუკეთესო ResNet18 checkpoint როგორც model artifact