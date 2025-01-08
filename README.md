# Documentation
Can be accessed [here](https://docs.google.com/document/d/1mGehrAw9oTr15yJbZZYTmkP1GT85za_Rvk9-iJZpD-Y/edit?usp=sharing).

# Setup and Installation Procedure

### Step 1: Clone the repository from GitHub
```bash
git clone https://github.com/MiaAmeen/myCNN
```

### Step 2: Activate PyVenv
```bash
python3 -m venv myenv
source myenv/bin/activate
```
Note: I use python version 3.12 to install requirements and run the project.

### Step 2: Install required packages with pip
```bash
cd myCNN
python3 -m pip install -r requirements.txt
```

# Options

### Arguments to run simple unsupervised KGE model:

- **`-f`, `--file_dir`** (default: `FILE_DIR`)
  - Description: Specify the directory containing the list of files to be parsed.
  - Example:
    ```bash
    python3 my_kge.py -f /path/to/files
    ```

- **`-t`, `--text_match`** (default: `None`)
  - Description: Provide text to match against graph nodes.
  - Example:
    ```bash
    python my_kge.py -t "search text"
    ```

- **`-n`, `--node_match`**
  - Description: A flag to match all graph nodes against each other. If included, this flag will be set to `True`. By default, it is `False`.
  - Example:
    ```bash
    python my_kge.py -n
    ```



