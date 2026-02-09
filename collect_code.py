import os

# Настройки: какие расширения собираем и какие папки игнорируем
INCLUDE_EXTENSIONS = {'.py', '.yml', '.yaml', '.Dockerfile', '.env.example'}
IGNORE_DIRS = {
    'venv', '.venv', 'env', '__pycache__', '.git', '.idea', '.vscode',
    'postgres_data', 'qdrant_storage', 'docs' # Игнорируем базу и документы
}
IGNORE_FILES = {'project_context.txt', 'collect_code.py'}

def collect_project_code(output_file="project_context.txt"):
    project_root = os.getcwd()
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        f_out.write(f"PROJECT STRUCTURE AND CODE\n")
        f_out.write(f"Generated: {os.path.abspath(output_file)}\n")
        f_out.write("="*50 + "\n\n")

        for root, dirs, files in os.walk(project_root):
            # Фильтруем папки (удаляем игнорируемые на месте)
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if file in IGNORE_FILES:
                    continue
                
                file_ext = os.path.splitext(file)[1]
                if file_ext in INCLUDE_EXTENSIONS or file in {'Dockerfile', 'requirements.txt'}:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, project_root)
                    
                    f_out.write(f"\nFILE: {relative_path}\n")
                    f_out.write("-" * (len(relative_path) + 6) + "\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f_in:
                            f_out.write(f_in.read())
                    except Exception as e:
                        f_out.write(f"Error reading file: {e}")
                    
                    f_out.write("\n" + "="*50 + "\n")

    print(f"✅ Контекст успешно собран в файл: {output_file}")

if __name__ == "__main__":
    collect_project_code()