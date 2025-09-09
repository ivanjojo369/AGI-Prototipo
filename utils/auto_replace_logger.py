import os
import re

PROJECT_DIR = "C:/Users/PC/Documents/AGI-Prototipo"
PATTERN = r'print\((.*?)\)'

def replace_print_with_logger(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    if 'Logger' not in content:
        content = 'from utils.logger import Logger\n' + content

    def replacer(match):
        message = match.group(1)
        msg_lower = message.lower()

        if "error" in msg_lower:
            return f'Logger.error({message})'
        elif "warning" in msg_lower:
            return f'Logger.warning({message})'
        elif "debug" in msg_lower:
            return f'Logger.debug({message})'
        else:
            return f'Logger.info({message})'

    new_content = re.sub(PATTERN, replacer, content)

    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        print(f"✅ Actualizado: {file_path}")

def process_directory(directory):
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.py') and filename not in ['logger.py', 'auto_replace_logger.py']:
                file_path = os.path.join(root, filename)
                replace_print_with_logger(file_path)

if __name__ == "__main__":
    print("🔄 Reemplazando print(...) por Logger.*(...) en todo el proyecto...")
    process_directory(PROJECT_DIR)
    print("✅ Finalizado. Todos los archivos usan Logger centralizado.")
