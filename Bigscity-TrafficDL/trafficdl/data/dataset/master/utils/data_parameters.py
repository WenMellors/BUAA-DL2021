data_parameters = {
    'taxi': {
        'data_train': "data/NYTaxi/taxi_train.npz",
        'data_val': "data/NYTaxi/taxi_val.npz",
        'data_test': "data/NYTaxi/taxi_test.npz",
        'data_type': 4,
        'data_name': [
            'inflow',
            'outflow',
            'in_transition',
            'out_transition'],
        'pred_type': 2,
        'data_max': [
            1409,
            1518,
            174,
            147],
        'len_int': 1800,
        'n_int': 48,
        'test_threshold': [
            10,
            10],
        'len_r': 16,
        'len_c': 12},
    'bike': {
        'data_train': "data/NYBike/bike_train.npz",
        'data_val': "data/NYBike/bike_val.npz",
        'data_test': "data/NYBike/bike_test.npz",
        'data_type': 4,
        'data_name': [
            'inflow',
            'outflow',
            'in_transition',
            'out_transition'],
        'pred_type': 2,
        'data_max': [
            262,
            274,
            39,
            33],
        'len_int': 1800,
        'n_int': 48,
        'test_threshold': [
            10,
            10],
        'len_r': 14,
        'len_c': 8},
    'ctm': {
        'data_train': "data/ctm/ctm_train.npz",
        'data_val': "data/ctm/ctm_val.npz",
        'data_test': "data/ctm/ctm_test.npz",
        'data_type': 2,
        'data_name': [
            'duration',
            'request'],
        'pred_type': 2,
        'data_max': [
            13151,
            2368],
        'len_int': 900,
        'n_int': 96,
        'test_threshold': [
            60,
            10],
        'len_r': 20,
        'len_c': 21}}
