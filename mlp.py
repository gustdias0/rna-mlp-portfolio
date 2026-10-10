import json
import os
import re
import unicodedata
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

FEATURES = ['Uf', 'IBGE', 'Municipio', 'CNES', 'Tipounidade', 'INE', 'Tipoequipe', 'Validacao', 'Total']
CATEGORICAL = ['Uf', 'Municipio', 'Tipounidade', 'Tipoequipe', 'Validacao']
REGIONS = {'CENTRO_OESTE': 0, 'NORDESTE': 1, 'NORTE': 2, 'SUDESTE': 3, 'SUL': 4}
SEED = 101


def clean_data(raw):
    data = raw.copy()
    data.columns = ['Regiao'] + FEATURES
    data = data.replace(r'^\s*$', np.nan, regex=True).dropna(subset=['Regiao'])
    for column in ['Regiao'] + CATEGORICAL:
        data[column] = data[column].fillna('AUSENTE').astype(str).map(
            lambda value: re.sub(r'[^A-Za-z0-9_]+', '_', unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode()).strip('_'))
    if not set(data.Regiao).issubset(REGIONS):
        raise ValueError('Regiao desconhecida na base.')
    for column in ['IBGE', 'CNES', 'INE']:
        data[column] = pd.to_numeric(data[column], errors='raise').fillna(-1)
    if (data.IBGE < 0).any():
        raise ValueError('IBGE ausente: nao e possivel separar municipios.')
    total = data.Total.astype(str)
    if not total.str.fullmatch(r'\d+|\d{1,3}(?:\.\d{3})+').all():
        raise ValueError('Total deve ser contagem inteira nao negativa; ponto separa milhares.')
    data['Total'] = total.str.replace('.', '', regex=False).astype('int64')
    return data.reset_index(drop=True)

def split_data(data):
    municipalities = data[['IBGE', 'Regiao']].drop_duplicates()
    if municipalities.IBGE.duplicated().any():
        raise ValueError('Mesmo IBGE associado a regioes diferentes.')
    train, holdout = train_test_split(municipalities, test_size=.4, stratify=municipalities.Regiao, random_state=SEED)
    val, test = train_test_split(holdout, test_size=.5, stratify=holdout.Regiao, random_state=SEED)
    return tuple(data.index[data.IBGE.isin(part.IBGE)].to_numpy() for part in [train, val, test])

def encode_data(data, train_index):
    encoded = data.copy()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int64)
    encoder.fit(data.loc[train_index, CATEGORICAL])
    encoded[CATEGORICAL] = encoder.transform(data[CATEGORICAL])
    encoded['Regiao'] = data.Regiao.map(REGIONS).astype('int64')
    scaler = MinMaxScaler().fit(encoded.loc[train_index, FEATURES])
    scaled = scaler.transform(encoded[FEATURES]).astype('float32')
    mappings = {column: {str(value): i for i, value in enumerate(categories)}
                for column, categories in zip(CATEGORICAL, encoder.categories_)}
    mappings['Regiao'] = REGIONS
    return encoded, scaled, scaler, mappings

def metric_intervals(real, predicted, groups, repetitions=1000):
    real, predicted, groups = np.asarray(real), np.asarray(predicted), np.asarray(groups)
    if len(real) != len(predicted) or len(real) != len(groups) or np.var(real) == 0:
        raise ValueError('Pares e grupos devem ter mesmo tamanho; alvo deve variar.')
    metrics = [mean_squared_error, mean_absolute_error, r2_score, explained_variance_score]
    point = [metric(real, predicted) for metric in metrics]
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    boot = []
    for _ in range(repetitions):
        indices = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        if np.var(real[indices]) > 0:
            boot.append([metric(real[indices], predicted[indices]) for metric in metrics])
    lower, upper = np.quantile(boot, [.025, .975], axis=0)
    return pd.DataFrame({'estimativa': point, 'ic95_inferior': lower, 'ic95_superior': upper}, index=['MSE', 'MAE', 'R2', 'EVS'])

def train_model(x, y, xv, yv, config):
    import tensorflow as tf
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    model = tf.keras.Sequential([tf.keras.Input(shape=(x.shape[1],)), tf.keras.layers.Dense(x.shape[1])]
                                + [tf.keras.layers.Dense(width, activation='relu') for width in config['layers']]
                                + [tf.keras.layers.Dense(1, activation='linear')])
    model.compile(optimizer=tf.keras.optimizers.Adam(config['learning_rate']), loss='mean_squared_error', metrics=['mean_squared_error'])
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    train = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=SEED).batch(config['batch_size']).with_options(options)
    validation = tf.data.Dataset.from_tensor_slices((xv, yv)).batch(config['batch_size']).with_options(options)
    history = model.fit(train, validation_data=validation, epochs=config['epochs'], verbose=0, shuffle=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', patience=15, restore_best_weights=True)])
    return model, history.history


def run_pipeline(raw, output, configs=None, repetitions=1000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, confusion_matrix

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = clean_data(raw)
    train, val, test = split_data(data)
    encoded, scaled, scaler, mappings = encode_data(data, train)
    y = encoded.Regiao.to_numpy(dtype='float32')
    configs = configs or [
        dict(layers=[100, 100], epochs=10, batch_size=64, learning_rate=.001),
        dict(layers=[64, 32], epochs=80, batch_size=64, learning_rate=.001),
        dict(layers=[100, 100], epochs=100, batch_size=64, learning_rate=.001)]
    experiments, histories = [], []
    best_model, best_value, best_id = None, np.inf, None
    for i, config in enumerate(configs):
        model, history = train_model(scaled[train], y[train], scaled[val], y[val], config)
        value = float(mean_squared_error(y[val], model(scaled[val], training=False).numpy().ravel()))
        experiments.append(dict(id=i + 1, **config, epochs_run=len(history['loss']), best_epoch=int(np.argmin(history['val_mean_squared_error'])) + 1, validation_mse=value))
        histories.append(history)
        print(f'Experimento {i + 1}: epocas={len(history["loss"])}, MSE validacao={value:.6f}', flush=True)
        if value < best_value:
            best_model, best_value, best_id = model, value, i

    predicted = best_model(scaled[test], training=False).numpy().ravel()
    intervals = metric_intervals(y[test], predicted, data.loc[test, 'IBGE'].to_numpy(), repetitions)
    classes = np.clip(np.rint(predicted), 0, 4).astype(int)
    test_states = data.loc[test, 'Uf'].to_numpy()
    unseen = encoded.loc[test, 'Uf'].to_numpy() == -1
    baseline = np.full(len(test), y[train].mean())
    missing = raw.replace(r'^\s*$', np.nan, regex=True).isna().sum()
    q1, q3 = data.Total.quantile([.25, .75])
    threshold = float(q3 + 1.5 * (q3 - q1))
    summary = dict(
        dataset='sintetico', seed=SEED, n=len(data), excluded=len(raw) - len(data),
        missing={str(k): int(v) for k, v in missing.items()},
        splits={name: dict(rows=len(index), municipalities=int(data.loc[index, 'IBGE'].nunique())) for name, index in [('treino', train), ('validacao', val), ('teste', test)]},
        classes={str(k): int(v) for k, v in data.Regiao.value_counts().items()},
        total=dict(min=int(data.Total.min()), max=int(data.Total.max()), median=float(data.Total.median()), mean=float(data.Total.mean()), q1=float(q1), q3=float(q3), threshold=threshold, outliers=int((data.Total > threshold).sum())),
        experiments=experiments, selected=best_id + 1, metrics=intervals.to_dict(orient='index'),
        accuracy=float(accuracy_score(y[test], classes)), baseline_mse=float(mean_squared_error(y[test], baseline)),
        unknown={column: int((encoded.loc[test, column] == -1).sum()) for column in CATEGORICAL},
        unseen_states={state: dict(rows=int((test_states == state).sum()), errors=int((classes[test_states == state] != y[test][test_states == state]).sum()), mse=float(mean_squared_error(y[test][test_states == state], predicted[test_states == state]))) for state in np.unique(test_states[unseen])},
        known_states_mse=float(mean_squared_error(y[test][~unseen], predicted[~unseen])) if (~unseen).any() else None,
        bootstrap_repetitions=repetitions)
    encoded.to_csv(output / 'Dados.csv', index=False)
    data.to_csv(output / 'Dados_limpos.csv', index=False)
    pd.DataFrame({'indice_original': test, 'IBGE': data.loc[test, 'IBGE'].to_numpy(), 'real': y[test], 'predito': predicted, 'classe_predita': classes}).to_csv(output / 'predicoes_teste.csv', index=False)
    partitions = pd.Series(index=data.index, dtype='str', name='particao')
    for name, index in [('treino', train), ('validacao', val), ('teste', test)]:
        partitions.loc[index] = name
    partitions.to_csv(output / 'particoes.csv', index_label='indice_original')
    intervals.to_csv(output / 'metricas_ic95.csv', index_label='metrica')
    pd.DataFrame(experiments).to_csv(output / 'experimentos.csv', index=False)
    (output / 'codificacao.json').write_text(json.dumps(mappings, indent=2), encoding='utf-8')
    (output / 'normalizacao.json').write_text(json.dumps({'features': FEATURES, 'scale': scaler.scale_.tolist(), 'min': scaler.min_.tolist()}, indent=2), encoding='utf-8')
    (output / 'historicos.json').write_text(json.dumps(histories, indent=2), encoding='utf-8')
    best_model.save(output / 'modelo.keras')
    (output / 'resultados.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    counts = data.Regiao.value_counts().reindex(REGIONS)
    axes[0, 0].bar(range(5), counts, color='#237a89')
    axes[0, 0].set_xticks(range(5), [r.replace('_', '\n') for r in REGIONS], fontsize=8)
    axes[0, 0].set(title='Frequência das classes', ylabel='Registros')
    axes[0, 1].hist(data.Total, bins=50, color='#237a89')
    axes[0, 1].set(title='Histograma de Total (escala original)', xlabel='Total', ylabel='Registros')
    axes[0, 1].locator_params(axis='x', nbins=5)
    axes[1, 0].hist(np.log1p(data.Total), bins=50, color='#237a89')
    axes[1, 0].set(title='Histograma de log(1 + Total)', xlabel='log(1 + Total)', ylabel='Registros')
    axes[1, 1].boxplot([data.loc[data.Regiao == r, 'Total'] for r in REGIONS], tick_labels=[r.replace('_', '\n') for r in REGIONS])
    axes[1, 1].tick_params(axis='x', labelsize=8)
    axes[1, 1].set(title='Dispersão e outliers por região', ylabel='Total (escala log)', yscale='log')
    fig.savefig(output / 'histogramas.png', dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4), layout='constrained')
    for i, history in enumerate(histories):
        epochs = np.arange(1, len(history['loss']) + 1)
        line, = ax.plot(epochs, history['mean_squared_error'], label=f'E{i + 1} treino')
        ax.plot(epochs, history['val_mean_squared_error'], '--', color=line.get_color(), label=f'E{i + 1} validacao')
    ax.axhline(.5, color='black', linestyle=':', label='Meta 0,5')
    ax.set(xlabel='Época', ylabel='MSE dos códigos de região', title='Evolução do MSE - treino e validação')
    ax.legend(ncol=4, fontsize=8)
    fig.savefig(output / 'evolucao_mse.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    order = np.argsort(y[test], kind='stable')
    sample = order[np.linspace(0, len(order) - 1, min(500, len(order)), dtype=int)]
    axes[0].plot(y[test][sample], '.', label='Real', color='#172b4d')
    axes[0].plot(predicted[sample], '.', alpha=.5, label='Predito contínuo', color='#d97632')
    axes[0].set(xlabel='Amostras de teste (mesma ordem para os pares)', ylabel='Código de região', title=f'Real x predito - até 500 de {len(test)} registros')
    axes[0].legend()
    cm = confusion_matrix(y[test], classes, labels=range(5))
    axes[1].imshow(cm, cmap='Blues')
    for i in range(5):
        for j in range(5):
            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')
    axes[1].set(title='Matriz de confusão (arredondamento)', xlabel='Classe predita', ylabel='Classe real', xticks=range(5), yticks=range(5))
    fig.savefig(output / 'real_predito.png', dpi=180)
    plt.close(fig)
    create_report(summary, output)
    print(intervals.to_string(), flush=True)
    return summary


def create_report(s, output):
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from reportlab.lib.pagesizes import A4
    output = Path(output)
    styles = getSampleStyleSheet()
    styles['Normal'].fontSize = 10
    styles['Normal'].leading = 14
    styles['Normal'].alignment = TA_JUSTIFY
    styles['Title'].textColor = colors.HexColor('#173b53')
    styles['Heading1'].textColor = colors.HexColor('#173b53')
    story = []
    def paragraph(text, style='Normal'):
        story.extend([Paragraph(text, styles[style]), Spacer(1, 8)])
    def table(rows, widths=None):
        t = Table([[Paragraph(str(cell), styles['Normal']) for cell in row] for row in rows], colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e3edf0')), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('BOTTOMPADDING', (0, 0), (-1, -1), 7), ('TOPPADDING', (0, 0), (-1, -1), 7), ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#cbd5dc'))]))
        story.extend([t, Spacer(1, 12)])
    def image(name, height):
        story.extend([Image(str(output / name), width=495, height=height), Spacer(1, 10)])

    paragraph('RNA-MLP | Demonstração', 'Title')
    paragraph('Dados sintéticos · Projeto para portfólio', 'Heading2')
    paragraph('Este projeto demonstra um fluxo de dados tabulares: limpeza, separação por grupos, treinamento de uma rede MLP e avaliação. Todos os registros foram gerados por código. Nenhum município, estabelecimento, equipe ou volume de produção corresponde a uma observação real.')
    mse = s['metrics']['MSE']['estimativa']
    paragraph(f'MSE de teste = {mse:.6f} | Acurácia após arredondamento = {s["accuracy"]:.2%}', 'Heading2')
    paragraph('1. Dados e protocolo', 'Heading1')
    paragraph(f'A base tem {s["n"]:,} registros. As categorias de região usam nomes conhecidos apenas como rótulos; as UFs XA-XJ são fictícias. Os identificadores IBGE, CNES e INE são marcadores sintéticos, sem vínculo com cadastros oficiais. A semente do gerador e do treinamento é 101.')
    paragraph('O gerador inclui classes desbalanceadas, valores ausentes e uma distribuição de Total com cauda à direita. Os números de identificação são aleatórios e independentes da região. Cada UF fictícia pertence a uma região por construção, tornando a relação aprendível e deliberadamente simples.')
    table([['Ausência original', 'Tratamento']] + [[f'{k}: {v:,}', 'Sentinela -1' if k == 'INE' else 'Categoria AUSENTE'] for k, v in s['missing'].items() if v], [245, 250])
    paragraph('Textos são padronizados sem acentos, espaços viram sublinhados e Total é lido como contagem inteira: 2.016 significa 2016. A codificação ordinal e a normalização MinMax são ajustadas apenas no treino. Categorias inéditas recebem -1. Regiao é excluída das nove entradas explicitamente.')
    table([['Partição', 'Registros', 'Municípios fictícios']] + [[k, v['rows'], v['municipalities']] for k, v in s['splits'].items()], [165, 165, 165])
    paragraph('A separação é feita por município, com 60% dos grupos para treino, 20% para validação e 20% para teste, estratificada por região. Todas as linhas de um município ficam juntas. A proporção de linhas varia com o tamanho de cada grupo.')
    story.append(PageBreak())

    paragraph('2. Distribuição dos dados', 'Heading1')
    image('histogramas.png', 346.5)
    table([['Região / código', 'Registros', 'Proporção']] + [[f'{r} / {REGIONS[r]}', n, f'{n/s["n"]:.2%}'] for r, n in s['classes'].items()], [245, 110, 140])
    t = s['total']
    paragraph(f'A classe dominante é {max(s["classes"], key=s["classes"].get)}. Total tem mediana {t["median"]:.1f}, média {t["mean"]:.2f} e máximo {t["max"]:,}. A regra Q3 + 1,5 IQR identifica {t["outliers"]:,} valores acima de {t["threshold"]:.1f}. Os outliers são preservados: fazem parte da distribuição simulada. O logaritmo é usado apenas na visualização.')
    story.append(PageBreak())

    paragraph('3. Treinamento', 'Heading1')
    paragraph('A rede recebe nove entradas: Uf, IBGE, Municipio, CNES, Tipounidade, INE, Tipoequipe, Validacao e Total. Uma Dense linear de nove unidades precede as camadas ocultas ReLU e a saída linear. O otimizador é Adam e a perda é MSE. A execução usa CPU e operações determinísticas.')
    table([['Teste', 'Ocultas', 'Épocas<br/>máx./rodadas', 'Melhor<br/>época', 'MSE<br/>validação']] + [[f'E{e["id"]}', str(e['layers']), f'{e["epochs"]}/{e["epochs_run"]}', e['best_epoch'], f'{e["validation_mse"]:.6f}'] for e in s['experiments']], [48, 112, 112, 85, 138])
    selected = s['experiments'][s['selected'] - 1]
    paragraph(f'E{s["selected"]} foi selecionado pelo menor MSE de validação. Batch = {selected["batch_size"]}; learning rate = {selected["learning_rate"]}. Early stopping tem paciência de 15 épocas e restaura os melhores pesos. O teste não participa da seleção de hiperparâmetros.')
    image('evolucao_mse.png', 198)
    paragraph('São comparadas duas larguras de rede e diferentes limites de épocas. Os gráficos e as tabelas vêm dos históricos efetivamente produzidos nesta execução; resultados anteriores não são reaproveitados.')
    paragraph('Regiao é nominal. A saída linear e as métricas de regressão foram preservadas da versão didática que originou o projeto. O MSE depende da ordem arbitrária dos códigos e não mede distância geográfica. Para um problema real de classificação, uma saída softmax e avaliação por classe seriam escolhas mais adequadas.')
    story.append(PageBreak())

    paragraph('4. Avaliação e limites', 'Heading1')
    image('real_predito.png', 198)
    table([['Métrica', 'Estimativa', 'IC95% inferior', 'IC95% superior']] + [[k, f'{v["estimativa"]:.6f}', f'{v["ic95_inferior"]:.6f}', f'{v["ic95_superior"]:.6f}'] for k, v in s['metrics'].items()], [90, 135, 135, 135])
    paragraph(f'Referência constante, média do alvo no treino: MSE = {s["baseline_mse"]:.6f}. A acurácia complementar é {s["accuracy"]:.2%}, calculada após arredondar e limitar as previsões a 0-4. As demais métricas usam previsões contínuas. O gráfico mantém os pares real/predito da mesma linha.')
    paragraph('Codificação do alvo: ' + '; '.join(f'{r} = {code}' for r, code in REGIONS.items()) + '.')
    paragraph(f'Os intervalos usam {s["bootstrap_repetitions"]:,} reamostragens dos municípios de teste com reposição, mantendo todas as linhas e seus pares em cada grupo. São os percentis 2,5% e 97,5%, condicionados ao modelo treinado e à divisão atual. Não medem a variação entre novos treinamentos.')
    paragraph('Categorias inéditas no teste: ' + '; '.join(f'{k}: {v}' for k, v in s['unknown'].items()) + '.')
    paragraph('Resultados sintéticos demonstram o funcionamento do pipeline. Não validam desempenho em dados reais, representatividade regional ou aplicação em saúde. A relação UF-região foi inserida pelo próprio gerador; um mapeamento direto poderia resolver essa tarefa sem rede neural.')
    paragraph('Reprodução', 'Heading2')
    paragraph('Execute python executar.py na raiz do projeto. O gerador recria data/dados_sinteticos.csv; o pipeline grava o modelo, os dados tratados, as partições, as métricas, os gráficos e este relatório em output/. O notebook Projeto_RNA_MLP.ipynb apresenta o mesmo fluxo. O README documenta o ambiente e os resultados medidos.')
    def footer(canvas, doc):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#526470'))
        canvas.drawString(50, 28, 'RNA-MLP | Dados sintéticos | Portfólio')
        canvas.drawRightString(A4[0] - 50, 28, str(doc.page))
    SimpleDocTemplate(str(output / 'Relatorio_RNA_MLP.pdf'), title='RNA-MLP - Dados sintéticos', author='', pagesize=A4, rightMargin=50, leftMargin=50, topMargin=42, bottomMargin=45).build(story, onFirstPage=footer, onLaterPages=footer)
