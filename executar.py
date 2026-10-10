"""Recria a demonstração completa com dados exclusivamente sintéticos."""

import json
from importlib.metadata import version
from pathlib import Path

from gerar_dados import generate_data
from mlp import run_pipeline


if __name__ == '__main__':
    root = Path(__file__).parent
    raw = generate_data()
    (root / 'data').mkdir(exist_ok=True)
    raw.to_csv(root / 'data' / 'dados_sinteticos.csv', index=False)
    (root / 'data' / 'origem.json').write_text(json.dumps({
        'synthetic': True, 'generator': 'gerar_dados.py', 'seed': 101,
        'rows': len(raw), 'municipalities': int(raw.IBGE.nunique())}, indent=2))
    result = run_pipeline(raw, root / 'output')
    packages = ['tensorflow', 'keras', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'reportlab']
    (root / 'output' / 'versoes_ambiente.json').write_text(
        json.dumps({name: version(name) for name in packages}, indent=2))
    print(f'Concluído: MSE de teste = {result["metrics"]["MSE"]["estimativa"]:.6f}')
