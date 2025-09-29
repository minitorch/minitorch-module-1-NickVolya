[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=20690476&assignment_repo_type=AssignmentRepo)
# MiniTorch Module 1

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module1/module1/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py tests/test_module.py tests/test_operators.py project/run_manual.py


# Task 1.5 results

## Dataset Simple:
```
PTS = 50
HIDDEN = 2
RATE = 0.5
data = minitorch.datasets["Simple"](PTS)
ScalarTrain(HIDDEN).train(data, RATE)
```

Logs:
```
Epoch  10  loss  34.82080718393398 correct 26
Epoch  20  loss  34.5554677409544 correct 26
Epoch  30  loss  34.13857007273073 correct 26
Epoch  40  loss  30.782478129772205 correct 29
Epoch  50  loss  24.389624725178916 correct 46
Epoch  60  loss  17.657655439792187 correct 46
Epoch  70  loss  13.379765852334003 correct 49
Epoch  80  loss  10.702318209346402 correct 49
Epoch  90  loss  17.162267712732785 correct 42
Epoch  100  loss  12.296679131471473 correct 44
Epoch  110  loss  11.889919350019852 correct 44
Epoch  120  loss  9.060819880257712 correct 45
Epoch  130  loss  7.741840948208283 correct 46
Epoch  140  loss  9.900794470411656 correct 45
Epoch  150  loss  7.5044925319300075 correct 46
Epoch  160  loss  6.736894512829931 correct 46
Epoch  170  loss  7.133860047741391 correct 46
Epoch  180  loss  5.804258173642894 correct 48
Epoch  190  loss  4.479155668640003 correct 48
Epoch  200  loss  4.043729116696729 correct 49
Epoch  210  loss  4.170601598081038 correct 48
Epoch  220  loss  11.474259716544037 correct 45
Epoch  230  loss  5.687222136775438 correct 47
Epoch  240  loss  2.6252791636878317 correct 50
Epoch  250  loss  2.3640166260556166 correct 50
Epoch  260  loss  2.1864687816530517 correct 50
Epoch  270  loss  2.0387032724334184 correct 50
Epoch  280  loss  1.8973988036209772 correct 50
Epoch  290  loss  1.7958878986376017 correct 50
Epoch  300  loss  1.6692551758118852 correct 50
Epoch  310  loss  1.563107128644895 correct 50
Epoch  320  loss  1.4794749838573267 correct 50
Epoch  330  loss  1.39928787287185 correct 50
Epoch  340  loss  1.32507188610293 correct 50
Epoch  350  loss  1.2578332784497042 correct 50
Epoch  360  loss  1.1960748893153155 correct 50
Epoch  370  loss  1.1391736929399436 correct 50
Epoch  380  loss  1.0863094473841475 correct 50
Epoch  390  loss  1.034846370558517 correct 50
Epoch  400  loss  0.9998282013765082 correct 50
Epoch  410  loss  0.9510154905227273 correct 50
Epoch  420  loss  0.9155791683370264 correct 50
Epoch  430  loss  0.8758907125585624 correct 50
Epoch  440  loss  0.846793120217233 correct 50
Epoch  450  loss  0.8117778905645217 correct 50
Epoch  460  loss  0.7862190292176725 correct 50
Epoch  470  loss  0.7551574176608696 correct 50
Epoch  480  loss  0.7309430931837939 correct 50
Epoch  490  loss  0.7057799687228844 correct 50
Epoch  500  loss  0.6808835333961445 correct 50
```


## Dataset Xor:
```
PTS = 50
data = minitorch.datasets["Xor"](PTS)
HIDDEN = 10
RATE = 0.5
ScalarTrain(HIDDEN).train(data, RATE)
```

Logs:
```
Epoch  10  loss  31.91960788602426 correct 36
Epoch  20  loss  29.51102694930734 correct 37
Epoch  30  loss  27.471020623812798 correct 37
Epoch  40  loss  25.46001704095667 correct 40
Epoch  50  loss  24.261429066027574 correct 39
Epoch  60  loss  24.5489307690804 correct 37
Epoch  70  loss  23.572941424056513 correct 37
Epoch  80  loss  22.08153192571325 correct 37
Epoch  90  loss  21.26366485446821 correct 38
Epoch  100  loss  21.632268078720216 correct 36
Epoch  110  loss  17.287612291323313 correct 43
Epoch  120  loss  19.21707994257873 correct 42
Epoch  130  loss  16.78703802035495 correct 42
Epoch  140  loss  13.858383037940776 correct 44
Epoch  150  loss  25.96031055817348 correct 34
Epoch  160  loss  11.376924132663042 correct 45
Epoch  170  loss  19.430781411357685 correct 41
Epoch  180  loss  10.395105513651057 correct 46
Epoch  190  loss  12.982874455884277 correct 46
Epoch  200  loss  11.370894553054239 correct 46
Epoch  210  loss  10.566218792288366 correct 46
Epoch  220  loss  16.363528793943466 correct 44
Epoch  230  loss  9.417868887944978 correct 46
Epoch  240  loss  19.485271172040544 correct 41
Epoch  250  loss  9.18726548128845 correct 47
Epoch  260  loss  12.964365738464108 correct 44
Epoch  270  loss  13.45785816205138 correct 45
Epoch  280  loss  10.293821214579705 correct 47
Epoch  290  loss  10.204588648573546 correct 47
Epoch  300  loss  9.926844202932378 correct 47
Epoch  310  loss  11.532850740438104 correct 46
Epoch  320  loss  12.9721036451376 correct 46
Epoch  330  loss  12.291506563079368 correct 45
Epoch  340  loss  11.311430313529803 correct 45
Epoch  350  loss  11.678392858349172 correct 45
Epoch  360  loss  10.75264370525952 correct 45
Epoch  370  loss  11.212185534182264 correct 45
Epoch  380  loss  10.536606888071525 correct 45
Epoch  390  loss  10.535144492739084 correct 47
Epoch  400  loss  10.671743524054538 correct 46
Epoch  410  loss  10.546826646366648 correct 46
Epoch  420  loss  10.044041089356856 correct 47
Epoch  430  loss  10.278603524049936 correct 46
Epoch  440  loss  9.796231337006802 correct 47
Epoch  450  loss  8.857885813650343 correct 47
Epoch  460  loss  7.252964772565461 correct 47
Epoch  470  loss  6.894384648122194 correct 47
Epoch  480  loss  6.948700674534503 correct 47
Epoch  490  loss  7.305635013996703 correct 47
Epoch  500  loss  6.635271185486863 correct 47
```


## Dataset Split
```
PTS = 50
HIDDEN = 10
RATE = 0.5
data = minitorch.datasets["Split"](PTS)
ScalarTrain(HIDDEN).train(data, RATE)
```

Logs:
```
Epoch  10  loss  31.929999096208554 correct 30
Epoch  20  loss  29.44405884025415 correct 34
Epoch  30  loss  26.323847382127095 correct 41
Epoch  40  loss  22.18705833069579 correct 46
Epoch  50  loss  19.611701869431844 correct 43
Epoch  60  loss  20.102325713931833 correct 39
Epoch  70  loss  20.496303595234856 correct 38
Epoch  80  loss  16.53625753508237 correct 41
Epoch  90  loss  20.3889518940579 correct 37
Epoch  100  loss  10.277511856025564 correct 46
Epoch  110  loss  16.40158727799402 correct 42
Epoch  120  loss  7.902687726997393 correct 48
Epoch  130  loss  5.924231977323139 correct 48
Epoch  140  loss  5.637019586399651 correct 49
Epoch  150  loss  58.728294967805354 correct 34
Epoch  160  loss  5.79989495460657 correct 48
Epoch  170  loss  4.890130231763578 correct 48
Epoch  180  loss  4.136347774529176 correct 48
Epoch  190  loss  4.38234280515873 correct 49
Epoch  200  loss  52.0231425525604 correct 32
Epoch  210  loss  5.185543590333229 correct 48
Epoch  220  loss  4.350965845774448 correct 49
Epoch  230  loss  3.8695478278823883 correct 49
Epoch  240  loss  3.5177069597960235 correct 49
Epoch  250  loss  3.0482741791567958 correct 49
Epoch  260  loss  8.976521650366747 correct 45
Epoch  270  loss  4.9224136678900745 correct 49
Epoch  280  loss  3.735779172326573 correct 49
Epoch  290  loss  3.3272559219281685 correct 49
Epoch  300  loss  3.051518441612005 correct 49
Epoch  310  loss  3.0056562677689675 correct 49
Epoch  320  loss  5.672377045177445 correct 49
Epoch  330  loss  4.990792626991642 correct 49
Epoch  340  loss  3.46815488122483 correct 49
Epoch  350  loss  29.636702596089428 correct 41
Epoch  360  loss  4.445926605001734 correct 48
Epoch  370  loss  3.30376173241082 correct 49
Epoch  380  loss  2.9059691769774343 correct 49
Epoch  390  loss  2.6536379264372236 correct 49
Epoch  400  loss  2.5111811461107303 correct 49
Epoch  410  loss  4.4485363720253 correct 49
Epoch  420  loss  3.1267728516604265 correct 49
Epoch  430  loss  3.8938379312433455 correct 49
Epoch  440  loss  3.6187100601307924 correct 49
Epoch  450  loss  3.709699354588776 correct 49
Epoch  460  loss  3.7323242626906334 correct 49
Epoch  470  loss  3.731129564968515 correct 49
Epoch  480  loss  3.747576755003776 correct 49
Epoch  490  loss  3.7469989904682746 correct 49
Epoch  500  loss  3.7461157994128436 correct 49
```


## Dataset Diag
```
PTS = 50
HIDDEN = 10
RATE = 0.1
data = minitorch.datasets["Diag"](PTS)
ScalarTrain(HIDDEN).train(data, RATE)
```

Logs:
```
Epoch  10  loss  15.575547444971114 correct 45
Epoch  20  loss  12.479792440648767 correct 45
Epoch  30  loss  11.542935223454396 correct 45
Epoch  40  loss  10.825189445325886 correct 45
Epoch  50  loss  10.156337735880445 correct 45
Epoch  60  loss  9.561518913565092 correct 45
Epoch  70  loss  8.975880243471691 correct 45
Epoch  80  loss  8.395362451444678 correct 45
Epoch  90  loss  7.8498734744132985 correct 45
Epoch  100  loss  7.327272799056724 correct 45
Epoch  110  loss  6.823144607538499 correct 46
Epoch  120  loss  6.342403482250929 correct 47
Epoch  130  loss  5.88905380609127 correct 47
Epoch  140  loss  5.4688939405495836 correct 48
Epoch  150  loss  5.076920575321984 correct 48
Epoch  160  loss  4.717154199738712 correct 50
Epoch  170  loss  4.389209393964371 correct 50
Epoch  180  loss  4.089338019260783 correct 50
Epoch  190  loss  3.8157256620381106 correct 50
Epoch  200  loss  3.5657445182921017 correct 50
Epoch  210  loss  3.3373473359824164 correct 50
Epoch  220  loss  3.1285055492179707 correct 50
Epoch  230  loss  2.9379291500789426 correct 50
Epoch  240  loss  2.76455430384754 correct 50
Epoch  250  loss  2.605811413959655 correct 50
Epoch  260  loss  2.4600955185621967 correct 50
Epoch  270  loss  2.326183336083038 correct 50
Epoch  280  loss  2.2027299496516566 correct 50
Epoch  290  loss  2.0887158287052396 correct 50
Epoch  300  loss  1.9833220672321608 correct 50
Epoch  310  loss  1.8857772211460404 correct 50
Epoch  320  loss  1.7956832700789802 correct 50
Epoch  330  loss  1.7121309932742088 correct 50
Epoch  340  loss  1.6343979104878337 correct 50
Epoch  350  loss  1.5617708446288985 correct 50
Epoch  360  loss  1.4939276650680757 correct 50
Epoch  370  loss  1.4304382032839327 correct 50
Epoch  380  loss  1.3709911052095078 correct 50
Epoch  390  loss  1.3151502789847986 correct 50
Epoch  400  loss  1.2626382541369228 correct 50
Epoch  410  loss  1.213334443466193 correct 50
Epoch  420  loss  1.1668726873721085 correct 50
Epoch  430  loss  1.1230403591589648 correct 50
Epoch  440  loss  1.0816863360397968 correct 50
Epoch  450  loss  1.0426156369150132 correct 50
Epoch  460  loss  1.0056372750588831 correct 50
Epoch  470  loss  0.9706087007271548 correct 50
Epoch  480  loss  0.9374418984906625 correct 50
Epoch  490  loss  0.9060438278055497 correct 50
Epoch  500  loss  0.8762170969933845 correct 50
```



