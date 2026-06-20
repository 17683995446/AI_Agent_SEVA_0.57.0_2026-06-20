#!/bin/bash
# SEVA Agent 安装脚本
# 用法: ./install.sh [install|develop|uninstall]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python
check_python() {
    print_info "检查Python环境..."
    if command -v python3 &> /dev/null; then
        PYTHON=python3
    elif command -v python &> /dev/null; then
        PYTHON=python
    else
        print_error "未找到Python，请安装Python 3.8+"
        exit 1
    fi

    VERSION=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    print_info "Python版本: $VERSION"

    # 检查版本 >= 3.8
    $PYTHON -c "
import sys
if sys.version_info < (3, 8):
    print('Python 3.8+ required')
    sys.exit(1)
"
    if [ $? -ne 0 ]; then
        print_error "需要Python 3.8+，当前版本: $VERSION"
        exit 1
    fi
}

# 检查pip
check_pip() {
    print_info "检查pip..."
    if ! $PYTHON -m pip --version &> /dev/null; then
        print_error "未找到pip，请安装pip"
        exit 1
    fi
    print_info "pip已安装"
}

# 安装依赖
install_deps() {
    print_info "安装依赖包..."
    $PYTHON -m pip install -r requirements.txt
    print_success "依赖安装完成"
}

# 安装SEVA
install_seva() {
    print_info "安装SEVA Agent..."
    $PYTHON -m pip install .
    print_success "SEVA Agent安装完成"
}

# 开发模式安装
install_dev() {
    print_info "以开发模式安装SEVA Agent..."
    $PYTHON -m pip install -e .
    print_success "SEVA Agent开发模式安装完成"
}

# 卸载SEVA
uninstall_seva() {
    print_info "卸载SEVA Agent..."
    $PYTHON -m pip uninstall -y seva-agent
    print_success "SEVA Agent已卸载"
}

# 验证安装
verify_install() {
    print_info "验证安装..."
    if $PYTHON -c "import seva; print(f'SEVA版本: {seva.__version__}')" 2>/dev/null; then
        print_success "SEVA Agent导入成功"
    else
        print_error "SEVA Agent导入失败"
        exit 1
    fi

    if command -v seva &> /dev/null; then
        print_success "seva命令可用"
        seva version
    else
        print_warning "seva命令未在PATH中，可能需要重新打开终端"
    fi
}

# 显示帮助
show_help() {
    echo "SEVA Agent 安装脚本"
    echo ""
    echo "用法: ./install.sh [命令]"
    echo ""
    echo "命令:"
    echo "  install     安装SEVA Agent（默认）"
    echo "  develop     以开发模式安装（可编辑）"
    echo "  uninstall   卸载SEVA Agent"
    echo "  help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./install.sh              # 安装"
    echo "  ./install.sh develop      # 开发模式安装"
    echo "  ./install.sh uninstall    # 卸载"
}

# 主函数
main() {
    local cmd=${1:-install}

    echo ""
    echo "========================================"
    echo "  SEVA Agent 安装程序"
    echo "========================================"
    echo ""

    case $cmd in
        install)
            check_python
            check_pip
            install_deps
            install_seva
            verify_install
            echo ""
            print_success "安装完成！使用 'seva --help' 开始使用"
            echo ""
            ;;
        develop|dev)
            check_python
            check_pip
            install_deps
            install_dev
            verify_install
            echo ""
            print_success "开发模式安装完成！"
            echo ""
            ;;
        uninstall)
            uninstall_seva
            echo ""
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $cmd"
            show_help
            exit 1
            ;;
    esac
}

main "$@"
