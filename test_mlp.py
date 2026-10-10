import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mlp


class DataTests(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame({
            'Região': ['CENTRO-OESTE', 'SUDESTE', None], 'Uf': ['XA', 'XB', 'XC'],
            'IBGE': ['9000001', '9000002', '9000003'],
            'Municipio': ['CIDADE ÁGUA', 'CIDADE NOVA', 'CIDADE TESTE'],
            'CNES': ['90000001', '90000002', '90000003'], 'Tipo Unidade': ['02', '', '02'],
            'INE': ['90000000001', None, '90000000003'], 'Tipo Equipe': ['70', '', '70'],
            'Validação': ['Preliminar>Aprovado'] * 3, 'Total': ['2.016', '646', '1']})

    def test_clean_preserves_thousands_and_treats_missing_without_losing_features(self):
        # Break caught: decimal parsing of Brazilian thousands, dropped required feature.
        result = mlp.clean_data(self.fixture())
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.columns.tolist(), ['Regiao', 'Uf', 'IBGE', 'Municipio', 'CNES', 'Tipounidade', 'INE', 'Tipoequipe', 'Validacao', 'Total'])
        self.assertEqual(result.Total.tolist(), [2016, 646])
        self.assertEqual(result.Municipio.tolist(), ['CIDADE_AGUA', 'CIDADE_NOVA'])
        self.assertEqual(result.Validacao.iloc[0], 'Preliminar_Aprovado')
        self.assertEqual(result.INE.iloc[1], -1)
        self.assertEqual(result.Tipounidade.iloc[1], 'AUSENTE')
        self.assertFalse(result.isna().any().any())

    def test_rejects_invalid_count_instead_of_silently_training_on_it(self):
        raw = self.fixture().iloc[:1].copy()
        raw.loc[0, 'Total'] = '-2'
        with self.assertRaises(ValueError):
            mlp.clean_data(raw)

    def test_split_keeps_municipalities_disjoint_and_all_regions_represented(self):
        # Break caught: row-wise split leaks municipality into the test partition.
        data = pd.DataFrame({'IBGE': np.repeat(np.arange(100), 3),
                             'Regiao': np.repeat(np.arange(100) % 5, 3)})
        result = mlp.split_data(data)
        self.assertIsInstance(result, tuple)
        train, val, test = result
        self.assertEqual([len(x) for x in result], [180, 60, 60])
        self.assertEqual(len(set(train) | set(val) | set(test)), 300)
        for a, b in [(train, val), (train, test), (val, test)]:
            self.assertFalse(set(data.loc[a, 'IBGE']) & set(data.loc[b, 'IBGE']))
        for index in result:
            self.assertEqual(data.loc[index, 'Regiao'].nunique(), 5)

    def test_encoding_learns_only_train_categories_and_excludes_target(self):
        # Break caught: fit on all rows or accidental target included in X.
        data = pd.DataFrame({
            'Regiao': ['NORTE', 'SUL', 'NORTE'], 'Uf': ['XA', 'XB', 'XX'],
            'IBGE': [1, 2, 3], 'Municipio': ['A', 'B', 'NOVA'], 'CNES': [1, 2, 3],
            'Tipounidade': ['02'] * 3, 'INE': [1, -1, 3],
            'Tipoequipe': ['70'] * 3, 'Validacao': ['OK'] * 3, 'Total': [10, 20, 1000]})
        result = mlp.encode_data(data, [0, 1])
        self.assertIsInstance(result, tuple)
        encoded, scaled, scaler, mappings = result
        self.assertEqual(scaled.shape, (3, 9))
        self.assertNotIn('Regiao', scaler.feature_names_in_)
        self.assertEqual(encoded.Uf.tolist(), [0, 1, -1])
        self.assertEqual(encoded.Regiao.tolist(), [2, 4, 2])
        self.assertEqual(float(scaled[2, -1]), 99.0)
        self.assertNotIn('XX', mappings['Uf'])


class TrainingTests(unittest.TestCase):
    def test_pipeline_exports_numeric_csv_paired_predictions_and_pdf(self):
        # Break caught: deliverables omitted or predictions detached from their test rows.
        rows = []
        for i in range(100):
            region = list(mlp.REGIONS)[i % 5].replace('_', '-')
            for j in range(2):
                rows.append([region, 'XX' if i == 11 else 'XA', str(100000 + i), f'CIDADE {i}', str(500000 + i),
                             '02', str(1000 + i), '70', 'Preliminar>Aprovado', str(j + 1)])
        raw = pd.DataFrame(rows, columns=DataTests().fixture().columns)
        with tempfile.TemporaryDirectory() as folder:
            result = mlp.run_pipeline(raw, folder, [dict(layers=[8], epochs=2, batch_size=32, learning_rate=.01)], repetitions=30)
            self.assertIsInstance(result, dict)
            self.assertIn('unseen_states', result)
            self.assertEqual(result['unseen_states']['XX']['rows'], 2)
            folder = Path(folder)
            data = pd.read_csv(folder / 'Dados.csv')
            self.assertEqual(data.shape, (200, 10))
            self.assertFalse(data.isna().any().any())
            self.assertTrue(all(pd.api.types.is_numeric_dtype(t) for t in data.dtypes))
            pred = pd.read_csv(folder / 'predicoes_teste.csv')
            expected = data.loc[pred.indice_original, 'Regiao'].to_numpy()
            np.testing.assert_array_equal(pred.real, expected)
            self.assertAlmostEqual(result['metrics']['MSE']['estimativa'], float(np.mean((pred.real - pred.predito)**2)), places=6)
            from pypdf import PdfReader
            reader = PdfReader(folder / 'Relatorio_RNA_MLP.pdf')
            text = '\n'.join(page.extract_text() for page in reader.pages)
            self.assertIn('Dados sintéticos', text)
            self.assertEqual(reader.metadata.author, '')
            self.assertNotIn('student', result)
            self.assertIn(f'{np.mean((pred.real - pred.predito)**2):.6f}', text)
            self.assertTrue((folder / 'modelo.keras').is_file())
            import tensorflow as tf
            restored = tf.keras.models.load_model(folder / 'modelo.keras')
            normalization = json.loads((folder / 'normalizacao.json').read_text())
            x = data.loc[pred.indice_original, mlp.FEATURES].to_numpy(dtype='float64')
            x = (x * normalization['scale'] + normalization['min']).astype('float32')
            np.testing.assert_allclose(restored(x, training=False).numpy().ravel(),
                                       pred.predito.to_numpy(), atol=1e-6, rtol=1e-6)
            for name in ['histogramas.png', 'evolucao_mse.png', 'real_predito.png']:
                self.assertGreater((folder / name).stat().st_size, 1000)

    def test_intervals_use_paired_predictions_and_resample_whole_groups(self):
        # Break caught: sorting real/predicted independently hides paired errors.
        real = np.array([0., 0., 2., 2.])
        pred = np.array([2., 2., 0., 0.])
        result = mlp.metric_intervals(real, pred, np.array([10, 10, 20, 20]), 60)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.loc['MSE', 'estimativa'], 4.0)
        self.assertEqual(result.loc['MAE', 'estimativa'], 2.0)
        self.assertEqual(result.loc['R2', 'estimativa'], -3.0)
        self.assertEqual(result.loc['EVS', 'estimativa'], -3.0)
        self.assertEqual(result.loc['MSE', 'ic95_inferior'], 4.0)
        self.assertEqual(result.loc['MSE', 'ic95_superior'], 4.0)

    def test_train_returns_one_linear_output_and_real_mse_history(self):
        x = np.linspace(0, 1, 120, dtype='float32').reshape(-1, 1)
        y = 2 * x[:, 0]
        config = dict(layers=[8], epochs=3, batch_size=16, learning_rate=.01)
        result = mlp.train_model(x[:90], y[:90], x[90:], y[90:], config)
        self.assertIsInstance(result, tuple)
        model, history = result
        self.assertEqual(model.output_shape[-1], 1)
        self.assertEqual(model.layers[-1].activation.__name__, 'linear')
        self.assertEqual(len(history['mean_squared_error']), 3)
        self.assertEqual(len(history['val_mean_squared_error']), 3)
        self.assertTrue(np.isfinite(model(x[90:], training=False).numpy()).all())


if __name__ == '__main__':
    unittest.main()
