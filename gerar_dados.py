"""Gera registros de demonstração sem ler qualquer fonte externa."""

from pathlib import Path

import numpy as np
import pandas as pd


def generate_data(seed=101):
    rng = np.random.default_rng(seed)
    regions = ['CENTRO-OESTE', 'NORDESTE', 'NORTE', 'SUDESTE', 'SUL']
    # Proporções arbitrárias para demonstrar classes desbalanceadas.
    labels = rng.permutation(np.repeat(np.arange(5), [60, 180, 60, 130, 70]))
    identifiers = rng.permutation(500) + 9000000
    rows = []
    for i, label in enumerate(labels):
        state = 'X' + chr(ord('A') + 2 * int(label) + int(rng.integers(2)))
        for _ in range(int(rng.integers(8, 25))):
            total = max(1, int(rng.lognormal(4.1, 1.25)))
            if rng.random() < .01:
                total *= 10
            rows.append([
                regions[label], state, str(identifiers[i]), f'MUNICÍPIO DEMO {i:04d}',
                '', rng.choice(['DEMO_A', 'DEMO_B', 'DEMO_C']), '',
                rng.choice(['EQUIPE_A', 'EQUIPE_B']), rng.choice(['PENDENTE', 'APROVADO']),
                f'{total:,}'.replace(',', '.')])
    data = pd.DataFrame(rows, columns=[
        'Região', 'Uf', 'IBGE', 'Municipio', 'CNES', 'Tipo Unidade',
        'INE', 'Tipo Equipe', 'Validação', 'Total'])
    # Faixas e prefixos de demonstração; não são códigos de cadastros oficiais.
    data['CNES'] = (rng.permutation(len(data)) + 90000000).astype(str)
    data['INE'] = (rng.permutation(len(data)) + 90000000000).astype(str)
    for column, rate in [('Tipo Unidade', .35), ('INE', .10), ('Tipo Equipe', .40)]:
        data.loc[rng.random(len(data)) < rate, column] = None
    return data


if __name__ == '__main__':
    target = Path(__file__).parent / 'data' / 'dados_sinteticos.csv'
    target.parent.mkdir(exist_ok=True)
    data = generate_data()
    data.to_csv(target, index=False)
    print(f'{len(data)} registros fictícios salvos em {target}')
