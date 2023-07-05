import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def make_fig(history):
    """
    Make figure from simple pipeline assessment.

    Show distributions for each operator with a combination of a boxplot and violinplot

    .. image:: DT_01.png
        :width: 500

    :param history: dict
        history from simple pipeline assessment

    :return: None
    """
    assert isinstance(history, dict), 'TypeError: history parameter should be an instance of dict'

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


def make_history_figs(history, mutations):
    """
        Make figure from simple pipeline assessment.

        Show distributions for each operator with subplots.

        Each subplot correspond to the average of a metric over all the iteration

        .. image:: history_figs.png
            :width: 500

        :param history: list
            histories from multiple mutations pipeline assessment

        :param mutations: list
            mutation ratios applied in pipeline

        :return: None
        """
    assert isinstance(history, list), 'TypeError: history parameter should be an instance of list'
    assert isinstance(mutations, list), 'TypeError: mutations parameter should be an instance of list'
    for element in mutations:
        assert isinstance(element, float), 'TypeError: mutations parameter elements should be instance of float'

    original = {'accuracy': [], 'dpd': [], 'eod': []}
    shuffle = {'accuracy': [], 'dpd': [], 'eod': []}
    killing = {'accuracy': [], 'dpd': [], 'eod': []}
    redistribution = {'accuracy': [], 'dpd': [], 'eod': []}
    new_class = {'accuracy': [], 'dpd': [], 'eod': []}
    datasets = [shuffle, killing, redistribution, new_class]
    names = ['shuffle', 'killing', 'redistribution', 'new_class']
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

    for i, df in enumerate(datasets):
        accuracy = df['accuracy']
        dpd = df['dpd']
        eod = df['eod']

        axes[i, 0].plot(accuracy)
        axes[i, 0].set_ylabel('Accuracy')
        axes[i, 0].set_xticks(range(len(mutations)))
        axes[i, 0].set_xticklabels(mutations)

        axes[i, 1].plot(dpd, color='r')
        axes[i, 1].set_ylabel('dpd')
        axes[i, 1].set_xticks(range(len(mutations)))
        axes[i, 1].set_xticklabels(mutations)

        axes[i, 2].plot(eod, color='g')
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

    plt.tight_layout()


def hist_to_dataframe(hist):
    """
    Convert history from basic pipeline to pandas dataFrame

    :param hist: dict
        basic pipeline result

    :return: DataFrame
        columns : metrics for each metrics
        rows : values over all iterations
    """
    assert isinstance(hist, dict), 'TypeError: hist parameter should be an instance of dict'
    # create empty list
    dfs = []
    # run through all operators
    for key in hist:
        # transpose dict to dataframe
        df = pd.DataFrame.from_dict(hist[key])
        # rename column with <operator>_<metric>
        new_columns = {}
        for col in df.columns:
            new_columns[col] = f'{key}_{col}'
        df.rename(columns=new_columns, inplace=True)
        dfs.append(df)

    # concat all dataframes in one
    result = pd.concat(dfs, axis=1)

    return result


def make_multi_hist_dataframe(histories, mutations):
    """
    Convert history from basic pipeline to pandas dataFrame

    :param histories: list
        histories returned from multiple mutations pipeline

    :param mutations: list
        mutation ration applied in pipeline

    :return: DataFrame
        columns : metrics for each metrics and mutation ratio
        rows : values over all iterations
    """
    # create empty list
    dfs = []

    # run through histories
    for i, hist in enumerate(histories):
        # convert history to dataframe
        df = hist_to_dataframe(hist)
        # append columns names with corresponding mutation ration
        new_columns = {}
        for col in df.columns:
            new_columns[col] = f'{mutations[i]}_{col}'
        df.rename(columns=new_columns, inplace=True)
        dfs.append(df)
    # concatenate dataframes along columns
    result = pd.concat(dfs, axis=1)

    return result
