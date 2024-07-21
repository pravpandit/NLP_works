### pip, conda source change

For more details, please move to [MirrorZ Help](https://help.mirrors.cernet.edu.cn/) to view.

#### pip source change

Temporarily use the mirror source to install, as shown below: `some-package` is the name of the package you need to install

```shell
pip install -i https://mirrors.cernet.edu.cn/pypi/web/simple some-package
```

Set the default mirror source for pip, upgrade pip to the latest version (>=10.0.0) and configure it as follows:

```shell
python -m pip install --upgrade pip
pip config set global.index-url https://mirrors.cernet.edu.cn/pypi/web/simple
```

If your pip default source has a poor network connection, temporarily use the mirror source to upgrade pip:

```shell
python -m pip install -i https://mirrors.cernet.edu.cn/pypi/web/simple --upgrade pip
```

#### conda source change

Mirror sites provide Anaconda repositories and third-party sources (conda-forge, msys2, pytorch, etc.). All systems can use mirror sites by modifying the .condarc file in the user directory.

The .condarc directories under different systems are as follows:

- Linux: ${HOME}/.condarc
- macOS: ${HOME}/.condarc
- Windows: C:\Users\<YourUserName>\.condarc

Note:

- Windows users cannot directly create a file named .condarc. You can execute conda config --set show_channel_urls yes to generate the file and then modify it.

Quick configuration

```shell
cat <<'EOF' > ~/.condarc
channels:
- defaults
show_channel_urls: true
default_channels:
- https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2 custom_channels: conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud EOF ````