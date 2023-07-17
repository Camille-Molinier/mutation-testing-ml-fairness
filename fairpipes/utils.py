import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pandas as pd


def make_fig(history, title=""):
    names = list(history.keys())[1:]
    accuracies = [history[key]['accuracy'] for key in names]
    dpd = [history[key]['dpd'] for key in names]
    eod = [history[key]['eod'] for key in names]
    fig, axes = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(hspace=0)

    sns.violinplot(data=accuracies, palette='turbo', inner=None, linewidth=0, saturation=0.4, ax=axes[0])
    sns.boxplot(data=accuracies, palette='turbo', width=0.3, boxprops={'zorder': 2}, ax=axes[0])
    axes[0].set_xticks(range(len(names)), names)
    axes[0].set_xticklabels(names, rotation=30)
    axes[0].set_ylabel('Accuracy')

    sns.violinplot(data=dpd, palette='turbo', inner=None, linewidth=0, saturation=0.4, ax=axes[1])
    sns.boxplot(data=dpd, palette='turbo', width=0.3, boxprops={'zorder': 2}, ax=axes[1])
    axes[1].set_xticks(range(len(names)), names)
    axes[1].set_xticklabels(names, rotation=30)
    axes[1].set_ylabel('dpd')

    sns.violinplot(data=eod, palette='turbo', inner=None, linewidth=0, saturation=0.4, ax=axes[2])
    sns.boxplot(data=eod, palette='turbo', width=0.3, boxprops={'zorder': 2}, ax=axes[2])
    axes[2].set_xticks(range(len(names)), names)
    axes[2].set_xticklabels(names, rotation=30)
    axes[2].set_ylabel('eod')


def make_history_figs(history, mutations, title=''):
    original = {'accuracy': [], 'dpd': [], 'eod': []}
    shuffle = {'accuracy': [], 'dpd': [], 'eod': []}
    killing = {'accuracy': [], 'dpd': [], 'eod': []}
    redistribution = {'accuracy': [], 'dpd': [], 'eod': []}
    new_class = {'accuracy': [], 'dpd': [], 'eod': []}
    datasets = [shuffle, killing, redistribution, new_class]
    names = ['column_shuffle', 'column_killing', 'redistribution', 'new_class']
    metrics = ['accuracy', 'dpd', 'eod']

    for mutation in history:
        for metric in metrics:
            original[metric].append(sum(mutation['original'][metric]) / len(mutation['original'][metric]))
            shuffle[metric].append(sum(mutation['column_shuffle'][metric]) / len(mutation['column_shuffle'][metric]))
            killing[metric].append(sum(mutation['column_dropping'][metric]) / len(mutation['column_dropping'][metric]))
            redistribution[metric].append(
                sum(mutation['redistribution'][metric]) / len(mutation['redistribution'][metric]))
            new_class[metric].append(sum(mutation['new_class'][metric]) / len(mutation['new_class'][metric]))

    fig, axes = plt.subplots(len(datasets), 3, figsize=(12, 8))

    original_acc = original['accuracy']
    original_dpd = original['dpd']
    original_eod = original['eod']

    for i, df in enumerate(datasets):
        accuracy = df['accuracy']
        dpd = df['dpd']
        eod = df['eod']

        axes[i, 0].plot(accuracy)
        axes[i, 0].plot(original_acc, '--', color='orange')
        axes[i, 0].set_ylabel('Accuracy')
        axes[i, 0].set_xticks(range(len(mutations)))
        axes[i, 0].set_xticklabels(mutations)

        axes[i, 1].plot(dpd, color='r')
        axes[i, 1].plot(original_dpd, '--', color='orange')
        axes[i, 1].set_ylabel('dpd')
        axes[i, 1].set_xticks(range(len(mutations)))
        axes[i, 1].set_xticklabels(mutations)

        axes[i, 2].plot(eod, color='g')
        axes[i, 2].plot(original_eod, '--', color='orange')
        axes[i, 2].set_ylabel('eod')
        axes[i, 2].set_xticks(range(len(mutations)))
        axes[i, 2].set_xticklabels(mutations)

        if i == len(datasets) - 1:
            axes[i, 0].set_xticks(range(len(mutations)))
            axes[i, 0].set_xticklabels(mutations)

            axes[i, 1].set_xticks(range(len(mutations)))
            axes[i, 1].set_xticklabels(mutations)

            axes[i, 2].set_xticks(range(len(mutations)))
            axes[i, 2].set_xticklabels(mutations)

        axes[i, 1].set_title(names[i])

    fig.suptitle(title)
    plt.tight_layout()


def hist_to_dataframe(hist):
    dfs = []
    for key in hist:
        df = pd.DataFrame.from_dict(hist[key])
        new_columns = {}
        for col in df.columns:
            new_columns[col] = f'{key}_{col}'
        df.rename(columns=new_columns, inplace=True)
        dfs.append(df)

    result = pd.concat(dfs, axis=1)
    return result


def make_multi_hist_dataframe(histories, mutations):
    dfs = []
    for i, hist in enumerate(histories):
        df = hist_to_dataframe(hist)
        new_columns = {}
        for col in df.columns:
            new_columns[col] = f'{mutations[i]}_{col}'
        df.rename(columns=new_columns, inplace=True)
        dfs.append(df)
    result = pd.concat(dfs, axis=1)
    return result


def make_stats(hist, display=True):
    dfs = []
    for key in list(hist.keys())[1:]:
        tmp = {'operator': key}
        for metric in hist[key]:
            _, p = mannwhitneyu(hist["original"][metric], hist[key][metric])
            tmp[metric] = [p]
        dfs.append(pd.DataFrame.from_dict(tmp))
    result = pd.concat(dfs)

    if display:
        pivot_table = result.pivot_table(index='operator', values=['accuracy', 'dpd', 'eod'])
        plt.figure()
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm', linewidths=0.5)

    return result


def make_multi_stats(hists, mutations):
    plt.figure(figsize=(15, 8))
    for i, hist in enumerate(hists):
        res = make_stats(hist, display=False)
        pivot_table = res.pivot_table(index='operator', values=['accuracy', 'dpd', 'eod'])
        plt.subplot(2, 3, i + 1)
        sns.heatmap(pivot_table, annot=True, cmap='coolwarm', linewidths=0.5, cbar=False)
        plt.title(f'mutation ratio = {mutations[i]}')

    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.tight_layout()
    plt.show()