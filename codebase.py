# codebase.py
import yaml
import csv
import io
from pathlib import Path


def load_yaml(yaml_path: str) -> dict:
    """Load and parse the codebase.yaml file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def extract_paths(data) -> list[str]:
    """
    Recursively extract all file paths from the yaml data.
    Supports multiple yaml formats:
      - flat list:     ['src/main.py', 'src/utils.py']
      - dict with key: {'files': ['src/main.py', ...]}
      - nested dict:   {'src': ['main.py', 'utils.py']}
    """
    paths = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                paths.append(item)
            else:
                paths.extend(extract_paths(item))

    elif isinstance(data, dict):
        for key, value in data.items():
            if value is None:
                paths.append(key)
            else:
                paths.extend(extract_paths(value))

    elif isinstance(data, str):
        paths.append(data)

    return paths


def resolve_paths(raw_paths: list[str], base_dir: Path) -> list[Path]:
    """
    Resolve each raw path string against the base directory.
    Returns only paths that actually exist on disk.
    Skips and warns about missing files.
    """
    resolved = []
    for raw in raw_paths:
        # normalize slashes, strip leading ./
        candidate = base_dir / Path(raw.strip())
        candidate = candidate.resolve()

        if candidate.exists():
            resolved.append(candidate.relative_to(base_dir))
        else:
            print(f"  ⚠️  Not found, skipping : {raw.strip()}")

    return resolved


def build_tree(relative_paths: list[Path]) -> dict:
    """
    Build a nested dict tree purely from resolved relative Path objects.
    Each node is a dict:  { name: {} }  for dirs,  { name: None }  for files.
    """
    tree = {}
    for path in relative_paths:
        parts = path.parts          # e.g. ('src', 'utils', 'helper.py')
        node  = tree
        for i, part in enumerate(parts):
            is_file = (i == len(parts) - 1)
            if is_file:
                node.setdefault(part, None)   # leaf → None
            else:
                node = node.setdefault(part, {})   # branch → dict
    return tree


def render_tree(tree: dict, prefix: str = "") -> list[str]:
    """Render the nested dict tree as pretty ASCII lines."""
    lines   = []
    entries = list(tree.items())

    for i, (name, subtree) in enumerate(entries):
        is_last   = (i == len(entries) - 1)
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{name}")

        if subtree:                                   # directory node
            extension = "    " if is_last else "│   "
            lines.extend(render_tree(subtree, prefix + extension))

    return lines


def get_language(file_path: Path) -> str:
    """Return markdown fence language tag for a given file extension."""
    extension_map = {
        '.py':    'python',
        '.js':    'javascript',
        '.ts':    'typescript',
        '.jsx':   'jsx',
        '.tsx':   'tsx',
        '.html':  'html',
        '.css':   'css',
        '.scss':  'scss',
        '.json':  'json',
        '.yaml':  'yaml',
        '.yml':   'yaml',
        '.md':    'markdown',
        '.sh':    'bash',
        '.bash':  'bash',
        '.sql':   'sql',
        '.xml':   'xml',
        '.toml':  'toml',
        '.ini':   'ini',
        '.env':   'bash',
        '.cpp':   'cpp',
        '.c':     'c',
        '.h':     'c',
        '.java':  'java',
        '.go':    'go',
        '.rs':    'rust',
        '.rb':    'ruby',
        '.php':   'php',
        '.swift': 'swift',
        '.kt':    'kotlin',
        '.r':     'r',
        '.dart':  'dart',
        '.csv':   'text',
    }
    return extension_map.get(file_path.suffix.lower(), 'text')


# ── CSV preview constant ───────────────────────────────────────────────────────
CSV_PREVIEW_ROWS = 5


def read_csv_preview(abs_path: Path) -> str:
    """
    Read a CSV file and return only the first CSV_PREVIEW_ROWS data rows
    (plus the header row if present), formatted as a plain CSV string.
    A truncation notice is appended when the file contains more rows.
    """
    try:
        raw_text = abs_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return "[Binary file — content not displayed]"
    except Exception as e:
        return f"[Error reading file: {e}]"

    reader     = csv.reader(io.StringIO(raw_text))
    all_rows   = list(reader)

    # Separate header (row 0) from data rows (row 1+)
    if not all_rows:
        return "[Empty CSV file]"

    header     = all_rows[:1]                        # always keep header
    data_rows  = all_rows[1:]                        # everything after header
    preview    = header + data_rows[:CSV_PREVIEW_ROWS]

    # Re-serialise rows back to CSV text so formatting is consistent
    output     = io.StringIO()
    writer     = csv.writer(output)
    writer.writerows(preview)
    result     = output.getvalue().rstrip('\r\n')

    # Append a notice if rows were trimmed
    total_data_rows = len(data_rows)
    if total_data_rows > CSV_PREVIEW_ROWS:
        remaining = total_data_rows - CSV_PREVIEW_ROWS
        result += (
            f"\n\n# ⚠️  Preview only — showing {CSV_PREVIEW_ROWS} of "
            f"{total_data_rows} data rows ({remaining} rows hidden)."
        )

    return result


def read_file(abs_path: Path) -> str:
    """
    Read and return file content as a string.
    CSV files are limited to the first 5 data rows to keep the output concise.
    """
    if abs_path.suffix.lower() == '.csv':
        return read_csv_preview(abs_path)

    try:
        return abs_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return "[Binary file — content not displayed]"
    except Exception as e:
        return f"[Error reading file: {e}]"


def generate_code_md(
    yaml_path:   str = "codebase.yaml",
    output_path: str = "CODE.md"
):
    base_dir = Path(__file__).parent.resolve()   # always the script's folder

    # ── 1. Load yaml & extract raw path strings ────────────────────
    print(f"\n📂 Loading '{yaml_path}' ...")
    yaml_data  = load_yaml(yaml_path)
    raw_paths  = extract_paths(yaml_data)
    print(f"   {len(raw_paths)} path(s) listed in yaml")

    # ── 2. Resolve paths dynamically from disk ─────────────────────
    print(f"\n🔍 Resolving paths against: {base_dir}")
    rel_paths = resolve_paths(raw_paths, base_dir)
    print(f"   {len(rel_paths)} file(s) found on disk")

    if not rel_paths:
        print("\n❌ No valid files to document. Exiting.")
        return

    # ── 3. Build tree from real relative paths ─────────────────────
    tree = build_tree(rel_paths)

    # ── 4. Compose CODE.md ─────────────────────────────────────────
    lines = []

    # Title
    lines += ["# Codebase", ""]

    # — Project Structure —
    lines += ["## 📁 Project Structure", "", "```", "."]
    lines += render_tree(tree)
    lines += ["```", ""]

    # — File Contents —
    lines += ["## 📄 File Contents", ""]

    for rel_path in rel_paths:
        abs_path = base_dir / rel_path
        language = get_language(rel_path)
        content  = read_file(abs_path)

        # Surface a hint in the terminal for CSV files
        suffix = rel_path.suffix.lower()
        label  = f" (preview: first {CSV_PREVIEW_ROWS} rows)" if suffix == '.csv' else ""
        print(f"  📝 {rel_path}{label}")

        # use forward slashes in the header regardless of OS
        lines += [
            f"### `{rel_path.as_posix()}`",
            "",
            f"```{language}",
            content,
            "```",
            "",
        ]

    # ── 5. Write output ────────────────────────────────────────────
    output_file = base_dir / output_path
    output_file.write_text('\n'.join(lines), encoding='utf-8')

    print(f"\n✅  '{output_path}' generated successfully!")
    print(f"    Files documented : {len(rel_paths)}")


if __name__ == "__main__":
    generate_code_md()