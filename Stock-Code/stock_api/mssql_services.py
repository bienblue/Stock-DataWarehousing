from sqlalchemy import create_engine, NVARCHAR, INTEGER, FLOAT
def determine_column_type(df, column_name):
    _x = df[column_name].dtype
    if df[column_name].dtype == 'O':  # Object type (string)
        return NVARCHAR()  # Adjust the length as per your requirement
    elif df[column_name].dtype == 'int64':
        return INTEGER()
    elif df[column_name].dtype == 'float64':
        return FLOAT()
    
def create_table(table_name, df):
    uri = 'mssql+pymssql://int-dba:vm1dta12#$@VN-PF3DT8K5/DW_STOCK'
    engine = create_engine(uri)
    connection = engine.connect()
    # Define a function to determine column types

# Create a dictionary to store column types
    column_types = {column: determine_column_type(df, column) for column in df.columns}

    # Insert the DataFrame into the MSSQL table with the specified column types
    df.to_sql(table_name, con=engine, if_exists='replace', index=False, dtype=column_types)

    # df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    connection.close()