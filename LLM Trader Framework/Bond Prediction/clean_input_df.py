import pandas as pd
from datetime import datetime, timedelta
import PyPDF2
from pathlib import Path
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl") 





def STRIPS_to_df(SHEET =0):
    """
    Reads STRIPS.xlsx file and cleans the data.
    
    Returns: pandas.DataFrame: Processed STRIPS data with datetime dates
    """
    
    # read Excel file for strips
    df = pd.read_excel('Input Data/STRIPS.xlsx', sheet_name=SHEET)  #IMPORTANT- 3 dif strips
    
    
    # remove empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

        # Keep only relevant cols
    df = df.iloc[:, :7]
    
    # Convert Date to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Set Date as index for time series analysis
    df.set_index('Date', inplace=True)
    
    return df


def STRIPS_120mins_to_df():
    """
    Reads STRIPS_120mins.xlsx file and cleans the data.
    
    Returns: pandas.DataFrame: Processed STRIPS 120-minute data with datetime index
    """
    import pandas as pd
    
    # Read the Excel file
    df = pd.read_excel('Input Data/STRIPS_120mins.xlsx', sheet_name=0) #IMPORTANT- selects bond tenor
    
    # Remove empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # Convert Date column to proper datetime objects
    # need to add year
    # 2024 MANUAL
    current_year = 2024
    df['Date'] = df['Date'].astype(str) + f'/{current_year}'
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d %H:%M/%Y')
    
    # Set Date as index for time series analysis
    df.set_index('Date', inplace=True)
    
    # Sort by date to ensure chronological order
    df = df.sort_index()
    
    return df




def yields_to_df():
    """
    Reads STRIPS.xlsx file and cleans the data.
    
    Returns:
        pandas.DataFrame: Processed STRIPS data with datetime dates
    """
    
    # Read the Excel file
    df = pd.read_excel('Input Data/term_structure.xlsx')  
    
    
    # Remove empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')

    
    # convert Date to  datetime objects
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # set Date as index for time series analysis
    df.set_index('Date', inplace=True)
    
    return df





def EFFR_to_df():
    """
    Reads EFFR.xlsx file and cleans the data up to Volume column.
    
    Returns:
        pandas.DataFrame: Processed EFFR data with datetime dates
    """
    
    # Read the Excel file
    df = pd.read_excel('Input Data/Target Rates.xlsx')
    
    # Keep only relevant cols
    df = df.iloc[:, :12]
    
    # Remove empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # Convert Effective Date to proper datetime objects
    df['Effective Date'] = pd.to_datetime(df['Effective Date'], format='%m/%d/%Y')
    
    # Effective Date as index for time series analysis
    df.set_index('Effective Date', inplace=True)
    
    return df



def Nowcast_to_df():
    """
    Reads Nowcast.csv file and cleans the data.
    
    Returns:
        pandas.DataFrame: Processed Nowcast data with datetime dates
    """
    
    # Read the CSV file
    df = pd.read_csv('Input Data/Nowcast historic.csv')
    
    # Remove empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # Convert Date to proper datetime objects
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Set Date as index for time series analysis
    df.set_index('Date', inplace=True)
    
    return df


def FOMC_PDFs_to_df():
    """
    Reads all PDF files from Input Data/FOMC Statements folder,
    extracts dates from filenames and content from PDFs.
    
    Returns:
        pandas.DataFrame: DataFrame with Date and Content columns
    """
    
    folder_path = Path('Input Data/FOMC Statements')
    dates = []
    contents = []
    pdf_files = list(folder_path.glob('*.pdf'))
    
    for pdf_file in pdf_files:
        filename = pdf_file.stem
        # Extract just the 8-digit date (monetary20240501a1 -> 20240501)
        date_str = ''.join([c for c in filename if c.isdigit()])[:8]
        
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            dates.append(date_obj)
            
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                contents.append(text_content.strip())
                
        except ValueError:
            continue
    
    df = pd.DataFrame({'Date': dates, 'Content': contents})
    df = df.sort_values('Date', ascending=False)
    df.set_index('Date', inplace=True)
    
    return df



#vol df (derived)
def vol_df_30_day(df):
    """
    Takes a DataFrame with Date index and Last Px column,
    returns a new DataFrame with 30-day rolling volatility of Last Px.
        
    Returns:
        pd.DataFrame: New DataFrame with Date index and '30D_Last_Vol' column
    """
    
    # Create a copy
    vol_df = df[['Last Px']].copy()

    #Sort so shift works
    vol_df = vol_df.sort_index()
    
    # Calculate daily returns (log returns for vol)
    vol_df['Last_Returns'] = np.log(vol_df['Last Px'] / vol_df['Last Px'].shift(1))
    
    # Calculate 30-day rolling standard deviation (volatility)
    vol_df['30D_Last_Vol'] = vol_df['Last_Returns'].rolling(window=30).std()
    
    # Annualise the volatility (multiply by sqrt(252) for trading days)
    vol_df['30D_Last_Vol'] = vol_df['30D_Last_Vol'] * np.sqrt(252)
    
    # Return only the date index and volatility column
    result_df = vol_df[['30D_Last_Vol']].copy()

    result_df = result_df.sort_index(ascending=False)
    
    return result_df






#basic benchmark df creation- allows use of most recent realised price as forward prediction

def STRIPS_forward_pricing():
    """
    Reads STRIPS.xlsx file and creates forward-looking pricing data.
    
    Returns:
        pandas.DataFrame: Data with Date, Last PX, Date+1, Last PX+1, Date+7, Last PX+7, Date+14, Last PX+14
    """
    import pandas as pd
    from datetime import timedelta
    
    # Read the Excel file using the same logic as original function
    df = pd.read_excel('Input Data/STRIPS.xlsx', sheet_name=2)
    
    # Remove empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # Keep only columns up to Volume (8th column)
    df = df.iloc[:, :7]
    
    # Convert Date to proper datetime objects
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    
    # Sort by date to ensure proper chronological order
    df = df.sort_values('Date')
    
    # Reset index to work with integer positions
    df = df.reset_index(drop=True)
    
    # Create the forward-looking dataframe
    forward_df = pd.DataFrame()
    
    # Add base date and price
    forward_df['Date'] = df['Date']
    forward_df['Last PX'] = df['Last Px']
    
    # Create forward-looking columns
    forward_df['Date + 1'] = None
    forward_df['Last PX + 1'] = None
    forward_df['Date + 7'] = df['Date'] + pd.Timedelta(days=7)
    forward_df['Last PX + 7'] = None
    forward_df['Date + 14'] = df['Date'] + pd.Timedelta(days=14)
    forward_df['Last PX + 14'] = None
    
    # Fill in the forward pricing data
    for i in range(len(df)):
        current_date = df.loc[i, 'Date']
        
        # Calculate target date for +1 day with Friday logic
        if current_date.weekday() == 4:  # Friday (0=Monday, 4=Friday)
            target_date_1 = current_date + timedelta(days=3)  # Friday -> Monday
        else:
            target_date_1 = current_date + timedelta(days=1)
        
        forward_df.loc[i, 'Date + 1'] = target_date_1
        
        # Find price 1 day ahead (with Friday logic)
        closest_1 = df[df['Date'] >= target_date_1]
        if not closest_1.empty:
            forward_df.loc[i, 'Last PX + 1'] = closest_1.iloc[0]['Last Px']
        
        # Find price 7 days ahead
        target_date_7 = current_date + pd.Timedelta(days=7)
        closest_7 = df[df['Date'] >= target_date_7]
        if not closest_7.empty:
            forward_df.loc[i, 'Last PX + 7'] = closest_7.iloc[0]['Last Px']
        
        # Find price 14 days ahead
        target_date_14 = current_date + pd.Timedelta(days=14)
        closest_14 = df[df['Date'] >= target_date_14]
        if not closest_14.empty:
            forward_df.loc[i, 'Last PX + 14'] = closest_14.iloc[0]['Last Px']

    # Remove last 10 rows
    forward_df = forward_df[:-10]
    
    # Save to CSV
    forward_df.to_csv('STRIPS_forward_pricing.csv', index=False)
    
    return forward_df



def aggregate_ARIMA(csv_1, csv_2, csv_3, name):
    """
    Reads 3 x arima output data from csv- creates clean csv for arima ensemble input
    """
    
    # Read the CSV files
    df_1 = pd.read_csv(csv_1)
    df_7 = pd.read_csv(csv_2)
    df_28 = pd.read_csv(csv_3)
    
    # Extract date and prediction columns, rename predictions
    merged = df_1[['date', 'prediction']].rename(columns={'prediction': 'pred_1'})
    merged = merged.merge(df_7[['date', 'prediction']].rename(columns={'prediction': 'pred_7'}), on='date', how='outer')
    merged = merged.merge(df_28[['date', 'prediction']].rename(columns={'prediction': 'pred_28'}), on='date', how='outer')
    
    # Sort by date and forward fill missing values
    merged = merged.sort_values('date')
    merged[['pred_7', 'pred_28']] = merged[['pred_7', 'pred_28']].fillna(method='ffill')
    
    # save as csv
    merged.to_csv(f'merged_arima_predictions_{name}.csv', index=False)



def ARIMA_to_df(treasury_name):
    """
    Takes merged arima data for given treasury name and converts to df
    """
    
    # Read the merged ARIMA predictions CSV
    df = pd.read_csv(f'Input Data/merged_arima_predictions_{treasury_name}.csv')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index for time series analysis
    df.set_index('date', inplace=True)
    
    return df





# helper functions - for modular use in trade pys
def get_last_n_days_data(df, current_date, n_days):
    """Get last n days of data from a dataframe with Date index"""
    end_date = pd.to_datetime(current_date)
    start_date = end_date - timedelta(days=n_days)
    
    # Filter data within date range
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    return filtered_df.to_string()

def get_actual_bond_price(strips_df, target_date):
    """Get actual bond price for a specific date"""
    try:
        target_dt = pd.to_datetime(target_date)
        if target_dt in strips_df.index:
            # last px used
            return strips_df.loc[target_dt, 'Last Px']
    except:
        pass
    return None










if __name__ == "__main__": #trigger prints for testing only
    # Example usage:
    strips_data = STRIPS_to_df()
    print(strips_data.tail(10))

    vol_data = vol_df_30_day(strips_data)
    print(vol_data.head(10))


    # Example usage:
    yield_data = yields_to_df()
    print(yield_data.head(10))

    # Example usage:
    rates_data = EFFR_to_df()
    print(rates_data.head(20))


    # Example usage:
    nowcast_data = Nowcast_to_df()
    print(nowcast_data.head())


    # Example usage:
    fomc_data = FOMC_PDFs_to_df()
    print(fomc_data.head())

    #print just the first columns values

    print("First date (index):", fomc_data.index[0])
    print("Full first content:", fomc_data.iloc[0]['Content'])

    strips_reformat = STRIPS_forward_pricing()
    print(strips_reformat.head())

    strips_120_data = STRIPS_120mins_to_df()
    print(strips_120_data.head(10))

    #aggregate arima
    aggregate_ARIMA("bond_predictions_ARIMA_SP_0_05_15_50_120min.csv","bond_predictions_ARIMA_SP_0_05_15_50_daily_7.csv","bond_predictions_ARIMA_SP_0_05_15_50_daily_28.csv","SP 0 05 15 50")

    # create aggregated ensemble arima dataset for given treasury
    arima_data = ARIMA_to_df("SP 0 05 15 50")
    print(arima_data.head(10))