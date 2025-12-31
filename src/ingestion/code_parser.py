import ast
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class CodeElement:
    """Represents a parsed code element."""
    type: str  # 'file', 'class', 'function'
    name: str
    content: str
    filepath: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    signature: Optional[str] = None
    complexity: int = 0
    imports: List[str] = None

    def __post_init__(self):
        if self.imports is None:
            self.imports = []


class CodeParser:
    """Parse Python code into structured elements."""

    def __init__(self, max_file_size_kb: int = 500):
        self.max_file_size_kb = max_file_size_kb

    def parse_repository(self, repo_path: str) -> List[CodeElement]:
        """Parse entire repository."""
        elements = []
        repo_path = Path(repo_path)

        logger.info(f"Parsing repository: {repo_path}")

        for py_file in repo_path.rglob("*.py"):
            # Skip large files and test files
            if self._should_skip_file(py_file):
                continue

            try:
                file_elements = self.parse_file(str(py_file))
                elements.extend(file_elements)
            except Exception as e:
                logger.warning(f"Failed to parse {py_file}: {e}")

        logger.info(f"Parsed {len(elements)} code elements")
        return elements

    def parse_file(self, filepath: str) -> List[CodeElement]:
        """Parse single Python file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            logger.warning(f"Syntax error in {filepath}: {e}")
            return []

        elements = []

        # File-level element
        file_element = self._create_file_element(filepath, source, tree)
        elements.append(file_element)

        # Class and function elements
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_element = self._create_class_element(filepath, source, node)
                elements.append(class_element)

                # Methods within class
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_element = self._create_function_element(
                            filepath, source, item, parent_class=node.name
                        )
                        elements.append(method_element)

            elif isinstance(node, ast.FunctionDef):
                # Top-level functions only
                if self._is_top_level(node, tree):
                    func_element = self._create_function_element(filepath, source, node)
                    elements.append(func_element)

        return elements

    def _create_file_element(self, filepath: str, source: str, tree: ast.AST) -> CodeElement:
        """Create file-level element."""
        # Extract module docstring
        docstring = ast.get_docstring(tree)

        # Extract imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        # Create summary (first 500 chars + docstring)
        summary = f"# File: {Path(filepath).name}\n"
        if docstring:
            summary += f'"""{docstring}"""\n\n'
        summary += f"Imports: {', '.join(imports[:10])}\n"
        summary += source[:500] + "..." if len(source) > 500 else source

        return CodeElement(
            type='file',
            name=Path(filepath).name,
            content=summary,
            filepath=filepath,
            start_line=1,
            end_line=len(source.splitlines()),
            docstring=docstring,
            imports=imports
        )

    def _create_class_element(self, filepath: str, source: str, node: ast.ClassDef) -> CodeElement:
        """Create class-level element."""
        docstring = ast.get_docstring(node)

        # Extract class signature and methods
        methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]

        # Get source code for this class
        lines = source.splitlines()
        class_source = '\n'.join(lines[node.lineno - 1:node.end_lineno])

        # Create summary
        signature = f"class {node.name}"
        if node.bases:
            bases = [ast.unparse(base) for base in node.bases]
            signature += f"({', '.join(bases)})"
        signature += ":"

        summary = f"{signature}\n"
        if docstring:
            summary += f'    """{docstring}"""\n'
        summary += f"\n    # Methods: {', '.join(methods)}\n"

        return CodeElement(
            type='class',
            name=node.name,
            content=class_source[:1000],  # Limit length
            filepath=filepath,
            start_line=node.lineno,
            end_line=node.end_lineno,
            docstring=docstring,
            signature=signature
        )

    def _create_function_element(
            self,
            filepath: str,
            source: str,
            node: ast.FunctionDef,
            parent_class: Optional[str] = None
    ) -> CodeElement:
        """Create function-level element."""
        docstring = ast.get_docstring(node)

        # Extract function signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        signature = f"def {node.name}({', '.join(args)})"
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"
        signature += ":"

        # Get source code
        lines = source.splitlines()
        func_source = '\n'.join(lines[node.lineno - 1:node.end_lineno])

        # Calculate complexity (simple metric: count control flow)
        complexity = self._calculate_complexity(node)

        # Create content
        content = f"{signature}\n"
        if docstring:
            content += f'    """{docstring}"""\n'
        content += func_source

        name = f"{parent_class}.{node.name}" if parent_class else node.name

        return CodeElement(
            type='function',
            name=name,
            content=content,
            filepath=filepath,
            start_line=node.lineno,
            end_line=node.end_lineno,
            docstring=docstring,
            signature=signature,
            complexity=complexity
        )

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _is_top_level(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is top-level (not nested in class)."""
        for item in tree.body:
            if item == node:
                return True
            if isinstance(item, ast.ClassDef):
                if node in item.body:
                    return False
        return False

    def _should_skip_file(self, filepath: Path) -> bool:
        """Check if file should be skipped."""
        # Skip test files
        if 'test' in filepath.name.lower():
            return True

        # Skip __pycache__ and virtual environments
        if any(part in str(filepath) for part in ['__pycache__', 'venv', '.venv', 'env']):
            return True

        # Skip large files
        if filepath.stat().st_size > self.max_file_size_kb * 1024:
            return True

        return False