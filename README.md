# Graduation Thesis
## Project name: Stock Data Warehousing
### Instructor: Master.Nguyen Huu Trung
## Tech used:
- MSSQL 
- SSIS 
- Superset 
- Streamlit
- Python, SQL

## Guide
### Get Data From Public API of TCBS and SSI 
- Inside folder stock-api of this project, which include the api-services.py and main.py. We were handle the API by json and numpy to convert it into csv file 
### Preprare LSTM model and Training model
- Export trading data from data warehouse to use for training and prediction
- Source code of model stored in main.py.
- After install neccesary librarys
Command to start appplication:
``` bash
streamlit run main.py
```
Open Web Browser and access :
http://localhost:8501

And finally, you can predict the stock price with current dataset or your own dataset.
But just keep in mind your data has same format with current dataset.

**Contact me if you have any problems while deploy website.**
_@ hongocbien0912@gmail.com_

