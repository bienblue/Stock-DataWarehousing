from sqlalchemy import create_engine, MetaData, Table, select, union_all
import requests
import json

MSSL_URI = 'mssql+pymssql://int-dba:vm1dta12#$@172.23.240.1/DW_STOCK'

engine = create_engine(MSSL_URI)
connection = engine.connect()
metadata = MetaData(bind=engine)

def generate_sql(_f_tb_name):
    _f_tb = Table(_f_tb_name, metadata, autoload=True)
    _f_tb_cols = [column.name for column in _f_tb.c]
    _dim_tables = []
    _selected_cols = []
    _join_conditions = []
    _selected_cols = []
    _index = 0
    for _fk in _f_tb.foreign_key_constraints:           
        _d_tb = _fk.referred_table
        _fc_key = _fk.column_keys[0]
        _dc_key = _fk.elements[0].column.name
        _d_columns = [f'd{_index}.{column.name} {_d_tb.name}_{column.name}' for column in _d_tb.c]

        _selected_cols.append(_d_columns)
        _join_conditions.append(f'f.{_fc_key} = d{_index}.{_dc_key}')
        _dim_tables.append(_d_tb.name)
        _index += 1

    _script = f'select \n\tf.*'
    if len(_dim_tables) != 0:
        _script += ',\n\t'
    else:
        _script += '\n'
    for i in range(len(_selected_cols)):
        _text_cols = ', '.join(str(_col) for _col in _selected_cols[i])
        _script += _text_cols
        if i == len(_selected_cols) - 1:
            _script += '\n'
        else:
            _script += ',\n\t'
    _script += f'from\n\t{_f_tb_name} f'

    for i in range(len(_dim_tables)):
        _script += f'\n\tleft join {_dim_tables[i]} d{i} ON {_join_conditions[i]}'


    with open('abc.sql', 'w') as file:
        file.write(_script)
    return _script


for _tb_name in engine.table_names():
    if 'fact' in _tb_name.lower():
        _script = generate_sql(_tb_name)
        _url = 'http://localhost:8080/api/v1/dataset/'
        headersList = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
            "Content-Length": "56",
            "Content-Type": "application/json",
            "Cookie": "SL_G_WPT_TO=vi; SL_GWPT_Show_Hide_tmp=undefined; SL_wptGlobTipTmp=undefined; session=.eJwljklqAzEQRe-idRYlqaQq-TJNjSTExNBtr4LvHkGWf3jwfsuRZ1yf5fY8X_FRji8vt8K1VojaQKj7WiSEE9PUbAm5eHYWc-9UG-8PspuodfQEgZHNdQI7DyabwZKAXWlp0N7WNF4OOmLWpssmMftgpaw20LSvBWWLvK44_23qjnadeTwf3_GzCx2daWr4RJNmQETQOqFToqYhuLUYbWzu_jC5x2Y2-P4D5pVEHw.ZH3i1Q.RKr6uziY3hfum30aXdU66FLJOuk",
            "Host": "localhost:8080",
            "Origin": "http://localhost:8080",
            "Referer": "http://localhost:8080/dataset/add/",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "same-origin",
            "Sec-Fetch-Site": "same-origin",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
            "X-CSRFToken": "ImI1Mzg3NmJlZDY0Y2EyYzA3NzcwMjM3NGQ3ZjRiZmM0MGRjMmU1MjUi.ZH3jQg.M4o1KigvPU09Hhz8r_uIgdloOSs",
            "sec-ch-ua": '"Google Chrome";v="113", "Chromium";v="113", "Not-A.Brand";v="24"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"'
        }
        payload = json.dumps({
            "database": 1,
            "external_url": None,
            "is_managed_externally": False,
            "sql": _script,
            "table_name": _tb_name
        })
        response = requests.post(_url, headers=headersList, data=payload)
        print(f'{_tb_name}: {response.status_code}')