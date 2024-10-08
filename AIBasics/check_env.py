import subprocess
import datetime


def check_packages(packages):
    """检查Python包是否安装"""
    for package in packages:
        try:
            __import__(package)
            print(f"{package} 已安装。")
        except ImportError:
            print(f"{package} 未安装。")
            install_package(package)


def install_package(package):
    if package in ['torch', 'pytorch']:
        print(f"{package} 需按照官网说明手动安装。")
    else:
        try:
            subprocess.run(['conda', 'install', package], check=True)
            print(f"Package {package} installed successfully.")
        except subprocess.CalledProcessError:
            try:
                subprocess.check_call(["pip", "install", package])
                print(f"{package} 安装成功。")
            except subprocess.CalledProcessError:
                print(f"{package} 安装失败。")


def main():
    # 需要检查的包列表
    packages = ['numpy', 'pandas', 'matplotlib', 'torch', 'scipy']
    check_packages(packages)
    print(datetime.datetime.now())


if __name__ == "__main__":
    main()
