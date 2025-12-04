### Environment Setup

```
conda create --name svrepair python=3.10 pip
pip install -r requirements.txt
```

### Download Data and Model
```
cd data
sh download_data.sh
```

```
cd checkpoint
sh download_model.sh
```

### Run
* Interface
```
sh run_test.sh
```
* Train
```
sh run_train.sh
```
* Static Analysis
```
sh run_code_analysis.sh
```