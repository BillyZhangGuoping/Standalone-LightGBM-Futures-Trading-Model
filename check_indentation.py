import re

def check_indentation(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines, 1):
        stripped_line = line.lstrip()
        if not stripped_line:
            continue  # skip empty lines
        indent = line[:len(line) - len(stripped_line)]
        if '\t' in indent:
            print(f"Line {i} uses tabs for indentation: {repr(line)}")
        else:
            indent_level = len(indent) // 4
            print(f"Line {i}: indent_level {indent_level}, content: {repr(line)}")

if __name__ == "__main__":
    check_indentation("backtest_trading_strategy.py")