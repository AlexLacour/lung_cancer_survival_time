from lifelines import CoxPHFitter
import pandas as pd
import metrics_t9gbvr2 as metric

if __name__ == '__main__':
    model = CoxPHFitter()
    X_data_dir = './x_train/features/clinical_data.csv'
    y_data_dir = './y_train/output.csv'
    rad_X_data_dir = './x_train/features/radiomics.csv'

    """
    ['shape', 'shape.1', 'shape.2', 'shape.3', 'shape.4', 'shape.5', 'shape.6', 'shape.7',
    'firstorder', 'firstorder.1', 'firstorder.2', 'firstorder.3', 'firstorder.4', 'firstorder.5',
    'firstorder.6', 'firstorder.7', 'firstorder.8', 'firstorder.9', 'firstorder.10', 'firstorder.11',
    'firstorder.12', 'firstorder.13',
    'textural', 'textural.1', 'textural.2', 'textural.3', 'textural.4', 'textural.5', 'textural.6',
    'textural.7', 'textural.8', 'textural.9', 'textural.10', 'textural.11', 'textural.12', 'textural.13',
    'textural.14', 'textural.15', 'textural.16', 'textural.17', 'textural.18', 'textural.19',
    'textural.20', 'textural.21', 'textural.22', 'textural.23', 'textural.24', 'textural.25',
    'textural.26', 'textural.27', 'textural.28', 'textural.29', 'textural.30']
    """

    X_data = pd.read_csv(X_data_dir, index_col='PatientID')
    y_data = pd.read_csv(y_data_dir, index_col='PatientID')
    rad_X_data = pd.read_csv(rad_X_data_dir, index_col='PatientID')
    print(list(rad_X_data.columns))
    rad_X_data = rad_X_data[['shape', 'shape.1', 'shape.2', 'shape.3', 'shape.4', 'shape.5', 'shape.6', 'shape.7',
                             'firstorder', 'firstorder.1', 'firstorder.2', 'firstorder.3', 'firstorder.4', 'firstorder.5',
                             'textural', 'textural.1', 'textural.2', 'textural.3', 'textural.4', 'textural.5', 'textural.6',
                             'textural.7', 'textural.8', 'textural.9', 'textural.10', 'textural.11', 'textural.12', 'textural.13',
                             'textural.14', 'textural.15', 'textural.16', 'textural.17', 'textural.18', 'textural.19',
                             'textural.20', 'textural.21', 'textural.22', 'textural.23', 'textural.24', 'textural.25',
                             'textural.26', 'textural.27', 'textural.28', 'textural.29', 'textural.30']]

    X_data = pd.concat([X_data, rad_X_data], axis=1)

    X_data = pd.get_dummies(X_data, drop_first=True)

    full_data = pd.concat([X_data, y_data], axis=1)

    full_data = full_data.dropna()

    print(full_data.head())

    model.fit(full_data, 'SurvivalTime', event_col='Event')
    model.print_summary()
    p = model.predict_expectation(X_data)
    print(p)

    p_df = pd.DataFrame(index=y_data.index)
    p_df['SurvivalTime'] = p
    p_df['Event'] = None
    p_df.SurvivalTime = p_df.SurvivalTime.fillna(p_df.SurvivalTime.mean())
    print(p_df.head())

    score = metric.cindex(y_data, p_df)
    print(f'CScore = {score}')
