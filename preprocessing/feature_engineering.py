
def get_nucleation_time(series_intensity, ignoreStartMinutes=10, nr_values_to_check=100):

    # todo: RAISE ERRORS
    nucleation_index = None
    nr_rows = series_intensity.shape[0]

    # Get differential
    series_diff = series_intensity.copy()
    for i in range(series_diff.shape[0]):
        value = series_diff.iloc[i]
        previous_value = series_intensity.iloc[i - 1] if i - 1 >= 0 else None
        diff = value - previous_value if previous_value is not None else 0.0
        series_diff.iloc[i] = diff
    df_diff = series_diff.rolling(int(nr_rows/10), min_periods=1, center=True).mean()

    # Check if last consecutive values all had positive differentials
    value_max_diff = 0
    for i in range(nr_rows):
        if nr_values_to_check < i < nr_rows - nr_values_to_check and df_diff.index[i].minute >= ignoreStartMinutes:
            current_diff_value = df_diff.iloc[i]
            previous_diff_values = df_diff.iloc[i-nr_values_to_check:i]
            next_diff_values = df_diff.iloc[i:i+nr_values_to_check]
            # Peak detection
            if df_diff.iloc[i] > value_max_diff and previous_diff_values.mean() < current_diff_value < next_diff_values.mean():
                value_max_diff = df_diff.iloc[i]
                nucleation_index = df_diff.index[i]

    # testing
    # import matplotlib.pyplot as plt
    # dfNucl = pd.DataFrame(data=[0 for _ in range(nr_rows)], columns=['nucleation'], index=df_diff.index.copy())
    # df_diff = pd.DataFrame(data=df_diff.copy(), columns=['diff'], index=df_diff.index.copy())
    # dfDf = pd.DataFrame(data=df.copy()/4000, columns=['value'], index=df_diff.index.copy())
    # dfNucl.loc[nuclIndex, 'nucleation'] = 0.001
    # dfJoined = df_diff.join(dfNucl)
    # dfJoined = dfJoined.join(dfDf)
    # dfJoined.plot()
    # plt.show()

    return nucleation_index
