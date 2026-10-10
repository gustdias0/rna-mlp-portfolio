import unittest

import numpy as np
import pandas as pd

from gerar_dados import generate_data
from mlp import clean_data, encode_data, split_data


class SyntheticDataTests(unittest.TestCase):
    def test_generated_data_is_reproducible_fictional_and_usable_by_the_pipeline(self):
        # Catches real-data reuse, unstable generation, or invalid grouped records.
        raw = generate_data()
        self.assertIsInstance(raw, pd.DataFrame)
        pd.testing.assert_frame_equal(raw, generate_data())
        self.assertFalse(raw.equals(generate_data(seed=202)))
        self.assertEqual(raw.shape[1], 10)
        self.assertGreater(len(raw), 5000)
        self.assertTrue(raw['Municipio'].str.startswith('MUNICÍPIO DEMO ').all())
        self.assertTrue(raw['Uf'].str.fullmatch('X[A-J]').all())
        self.assertEqual(raw['IBGE'].nunique(), 500)
        self.assertTrue(pd.to_numeric(raw['IBGE']).between(9000000, 9000499).all())
        self.assertTrue(pd.to_numeric(raw['CNES']).ge(90000000).all())
        self.assertTrue(raw['INE'].isna().any())
        self.assertTrue(raw['Tipo Unidade'].isna().any())
        self.assertTrue(raw['Tipo Equipe'].isna().any())
        self.assertTrue(raw['Total'].str.contains('.', regex=False).any())
        data = clean_data(raw)
        self.assertEqual(len(data), len(raw))
        self.assertFalse(data.isna().any().any())
        self.assertTrue((data.Total > 0).all())
        self.assertEqual(data.groupby('IBGE').Regiao.nunique().max(), 1)
        train, val, test = split_data(data)
        for a, b in [(train, val), (train, test), (val, test)]:
            self.assertFalse(set(data.loc[a, 'IBGE']) & set(data.loc[b, 'IBGE']))
        encoded, scaled, scaler, mappings = encode_data(data, train)
        self.assertEqual(scaled.shape, (len(raw), 9))
        self.assertTrue(np.isfinite(scaled).all())
        self.assertTrue(set(data.loc[test, 'Uf']).issubset(data.loc[train, 'Uf']))
        self.assertNotIn('Regiao', scaler.feature_names_in_)


if __name__ == '__main__':
    unittest.main()
