"""Сборка ноутбука по всем моделям индивидуального задания."""

from pathlib import Path
import textwrap
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def md(source):return nbformat.v4.new_markdown_cell(textwrap.dedent(source).strip())
def code(source):return nbformat.v4.new_code_cell(textwrap.dedent(source).strip())


cells=[
md('''
# Noisy-DQN, Noisy-A3C и Noisy-PPO

Курсовая работа по МГМО. Горбунов Иван Олегович, ИУ6-74б.
Руководитель — Кудрявцев Сергей Александрович.

Ноутбук закрывает модели индивидуального задания: Noisy-DQN без ε-жадного
выбора, Noisy-A3C, Noisy-PPO, независимый и факторизованный шум, динамику σ,
скорость и стабильность обучения. Подробная серия DQN сохранена в
`02_final_study.ipynb`. Здесь подробно проверяется дополнительная серия A3C/PPO.

Всего без пилота: 200 обучений, 26 млн переходов. Из них 120 обучений и 15 млн
переходов относятся к A3C/PPO. Ни один запуск не исключается по награде.
'''),
code('''
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from IPython.display import Image, Markdown, display
from noisynets.policy_analysis import load_policy_study, summarize_policy, make_figures, verify_policy, LABELS
from noisynets.actorcritic import PolicyConfig, create_model
from noisynets.policy_networks import PolicyNoisyLinear

ROOT=Path.cwd()
if not (ROOT/'configs/policy_study.json').exists(): ROOT=ROOT.parent
assert (ROOT/'configs/policy_study.json').exists()
OUT=ROOT/'results/policy_analysis';OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
pd.set_option('display.max_rows',100)
pd.set_option('display.float_format',lambda x:f'{x:.3f}')
runs,curves,manifest=load_policy_study(ROOT/'results/policy',ROOT/'configs/policy_study.json')
summary=summarize_policy(runs)
assert len(runs)==120 and summary['training_steps']==15_000_000
print('120 завершенных обучений, 15 000 000 переходов дополнительной серии')
'''),
md('''
## 1. Протокол

До запусков зафиксирован `configs/policy_study.json`. В каждой среде сравниваются
A3C, Noisy-A3C с независимым и факторизованным шумом, PPO и два Noisy-PPO.
Используются seed 200–209. Валидация — 10 эпизодов на seed 55000–55009,
отложенная оценка — 30 эпизодов на 65000–65029. Чекпойнт всегда последний.

Основной режим — sampling действий из categorical-политики. У шумовых моделей
ε обновляется каждые 20 шагов оценки A3C или 128 шагов PPO. Дополнительно
проверяются sampling и argmax по средним весам. Это разные политики.
'''),
code('''
display(pd.DataFrame([{ 'Среда':e['config']['env'],'Бюджет':e['config']['steps'],
    'Скрытый слой':e['config']['hidden'],'Число условий':len(e['conditions']),
    'σ₀':e['config']['sigma0'],'Потоков/сред':e['config']['workers']}
    for e in manifest['environments']]))
display(pd.DataFrame(manifest['environments'][0]['conditions']))
'''),
md(r'''
## 2. Что реализовано

**DQN** обучает Q-функцию по replay и target network. Его поведение ε-жадное.
В **Noisy-DQN** выбор действия жадный относительно возмущённой Q-функции,
а параметры μ и σ обучаются совместно. Исходные 80 запусков не изменены.

**A3C** использует четыре конкурентных actor-learner с собственными средами
и локальными сетями. После копирования общих параметров каждый независимо
собирает rollout до 20 шагов, строит n-step-возврат и градиент actor/critic.
Только копирование и применение Adam атомарны. Пока поток считает градиент,
другой может обновить общую сеть. Это регистрируется как staleness.
Синхронизация всех потоков выполняется только на границах оценки.

При истинном окончании bootstrap нулевой. При TimeLimit используется ценность
последнего наблюдения до reset. Actor использует преимущество G−V, отсоединённое
от графа; critic минимизирует половину квадрата ошибки. У базового A3C
есть entropy bonus β=0,01, у Noisy-A3C β=0. Sampling действий сохраняется.
В Noisy-A3C одна реализация шума фиксирована на rollout и обратный проход.

**PPO** собирает данные четырёх сред, затем выполняет четыре эпохи по minibatch64.
Применяются GAE(λ=0,95), clip0,2 и ограничение нормы градиента0,5.
В **Noisy-PPO** старая и текущая вероятность условны на одинаковом ε:

\[
\rho_t(\theta,\varepsilon)=
\frac{\pi_\theta(a_t\mid s_t,\varepsilon)}
{\pi_{old}(a_t\mid s_t,\varepsilon)}.
\]

Распределение ε не зависит от θ, поэтому сокращается в совместном отношении.
Шум не пересэмплируется до завершения всех эпох данного rollout. До первого
обновления код проверяет log-ratio≈0. Коэффициенты μ и σ при оптимизации меняются.
Noisy-PPO — описанная здесь интеграция для курсовой, а не модель из статьи2018г.
'''),
code('''
torch.manual_seed(82);a=create_model(PolicyConfig(),4,2)
torch.manual_seed(82);b=create_model(PolicyConfig(algorithm='noisy-ppo'),4,2)
a.eval();b.eval();x=torch.randn(10,4)
for left,right in zip(a(x),b(x)): torch.testing.assert_close(left,right)
print('Средние начальные параметры baseline/noisy совпадают')
display(b)
'''),
md('''
## 3. Два способа генерации шума

Независимый Gaussian-шум задаёт отдельный N(0,1) для каждого веса и смещения:
pq+q случайных значений. Факторизованный вариант использует p+q значений,
преобразование sign(x)√|x| и внешнее произведение. Ранг1 относится к ε весов,
а не обязательно к σ⊙ε. Равный σ₀ не означает одинаковую дисперсию возмущения.
'''),
code('''
rows=[]
for generation in ['independent','factorized']:
    layer=PolicyNoisyLinear(40,30,generation=generation)
    layer.reset_noise(torch.Generator().manual_seed(44))
    rows.append({'Генератор':generation,'Ранг ε':int(torch.linalg.matrix_rank(layer.weight_epsilon)),
                 'Среднее ε':float(layer.weight_epsilon.mean()),'SD ε':float(layer.weight_epsilon.std())})
display(pd.DataFrame(rows))
'''),
md('''
## 4. Качество и динамика обучения

Единица наблюдения — обучение с отдельным seed. Итог каждой модели — среднее
30 отложенных эпизодов. Среднее и SD ниже рассчитаны по десяти моделям.
AUC — трапецеидальная площадь по 11 валидационным точкам, делённая на бюджет.
Среда CartPole имеет положительный возврат, Acrobot — отрицательный; их шкалы
не усредняются. Полоса на графике — одно SD, не доверительный интервал.
'''),
code('''
conditions=pd.DataFrame(summary['conditions'])
display(conditions[['environment','label','final_mean','final_seed_sd','median','auc_mean','success_rate']])
make_figures(runs,curves,OUT/'figures')
for env in ['cartpole','acrobot']: display(Image(filename=str(OUT/'figures'/f'{env}_learning.png')))
'''),
md('''
## 5. Парные сравнения и генераторы

Все интервалы описательные, без поправки на множественные сравнения:
20 000 парных bootstrap-перевыборок, seed7241. Пары сохраняют общий seed
и начальные μ, но не общий опыт. В A3C дополнительно различается порядок потоков.
Нулевое значение внутри интервала не доказывает эквивалентности.
'''),
code('''
comparisons=pd.DataFrame(summary['comparisons'])
display(comparisons[['environment','condition','reference','metric','mean_difference','ci95_low','ci95_high']])
display(pd.DataFrame(summary['generation_comparisons'])[['environment','family','metric','mean_difference','ci95_low','ci95_high']])
'''),
md('''
## 6. Скорость достижения порога и стабильность

Пороги определены до запусков: validation475 для CartPole, −100 для Acrobot.
Недостижение — цензурированное наблюдение. Медиана времени среди достигших
показывается вместе с числом достигших, поэтому быстрый успех одного seed
не трактуется как успех всего метода. Это критерий по сетке оценок, а не
математическое доказательство сходимости. SD последних пяти оценок описывает
колебания внутри запуска, включая случайность контрольных эпизодов.
'''),
code('''
display(conditions[['environment','label','threshold_reached',
                   'threshold_step_median_among_reached','tail_validation_sd','final_seed_sd']])
'''),
md('''
## 7. Масштаб шума и режим применения модели

В σ не заложено принудительное затухание. Показана фактическая динамика
среднего модуля весовых коэффициентов, среднее и SD по десяти seed.
Масштабы смещений также сохранены в исходных CSV. Изменение σ не измеряет
напрямую исследованное пространство состояний. В режиме eval шум отключается.
'''),
code('''
display(Image(filename=str(OUT/'figures/sigma_weights.png')))
display(conditions[['environment','label','final_mean','mean_sampled','mean_greedy']])
display(pd.DataFrame(summary['mode_comparisons'])[['environment','condition','mean_difference','ci95_low','ci95_high']])
async_rows=[];ratio_errors=[]
for marker in sorted((ROOT/'results/policy').glob('*/*/seed_*/COMPLETE')):
    meta=json.loads((marker.parent/'metadata.json').read_text())
    if 'max_staleness' in meta: async_rows.append(meta['max_staleness'])
    if 'max_initial_logratio_error' in meta: ratio_errors.append(meta['max_initial_logratio_error'])
assert len(async_rows)==60 and min(async_rows)>0
assert len(ratio_errors)==60 and max(ratio_errors)<2e-5
print('A3C: у всех60 запусков есть асинхронные устаревшие градиенты; max=',max(async_rows))
print('PPO: максимальная ошибка начального log-ratio:',max(ratio_errors))
'''),
md('''
## 8. Повторная оценка всех сохранённых моделей

Сравниваются все награды, длины, seed и признаки terminated/truncated.
Проверяются120 checkpoint и9600 эпизодов во всех предусмотренных режимах.
Это проверка оценки, а не повторное обучение асинхронных моделей.
'''),
code('''
verification=verify_policy(ROOT/'results/policy')
verification.to_csv(OUT/'checkpoint_verification.csv',index=False)
assert len(verification)==320 and verification.n_episodes.sum()==9600
display(verification.groupby(['environment','mode'],as_index=False).agg(
    моделей=('matches','size'),эпизодов=('n_episodes','sum'),совпали=('matches','all')))
print('Все120 checkpoint и9600 эпизодов совпали')
'''),
md('''
## 9. Связь с серией DQN и ограничения

Серия DQN использует seed100–109 и иной протокол оценки. В ней Noisy-DQN
не показал превосходства по средним весам; на Acrobot включение шума резко
изменяет результат. Детальная статистика доступна в предыдущем ноутбуке.
Показатели разных семейств не образуют общий парный тест и не усредняются.

Noisy-A3C/PPO отличаются от baseline и шумовыми слоями, и отключённым
энтропийным бонусом. Поэтому сравнивается определённый вариант алгоритма,
а отдельное причинное влияние каждого компонента не выделено.
Обучение A3C не является побитово воспроизводимым из-за порядка потоков;
фиксированные checkpoint оцениваются воспроизводимо. Оценка использует
одну последовательность случайного шума на seed старта. Две среды с плотной
наградой не позволяют обобщить вывод на Atari и редкие события.
'''),
code('''
dqn=pd.read_csv(ROOT/'results/final_analysis/conditions.csv')
display(dqn.loc[dqn.condition.isin(['dqn','noisy']),
               ['environment','condition','final_mean','final_seed_sd','stochastic_mean']])
conclusions=[]
for row in summary['comparisons']:
    if row['metric']!='final_mean':continue
    outcome=('интервал целиком положителен' if row['ci95_low']>0 else
             'интервал целиком отрицателен' if row['ci95_high']<0 else 'интервал включает ноль')
    conclusions.append(f"{row['environment']}, {LABELS[row['condition']]}: Δ={row['mean_difference']:.2f}, "
                       f"95%[{row['ci95_low']:.2f};{row['ci95_high']:.2f}]; {outcome}.")
display(Markdown((chr(10)*2).join(conclusions)))
'''),
md('''
## Источники

1. Fortunato et al. [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295).
2. Mnih et al. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783).
3. Schulman et al. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
4. Schulman et al. [Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
5. Farama Foundation. [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/),
   [Acrobot](https://gymnasium.farama.org/environments/classic_control/acrobot/).
''')]

if __name__=='__main__':
    nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{
        'display_name':'Python3','language':'python','name':'python3'}})
    for cell in cells:
        if cell.cell_type=='code':compile(cell.source,'notebook','exec')
    path=ROOT/'notebooks/03_all_models.ipynb';nbformat.write(nb,path);print(path)
