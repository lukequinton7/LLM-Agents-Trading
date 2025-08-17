import numpy as np
import pandas as pd
import clean_input_df as input_
import matplotlib.pyplot as plt


def cap_df(date_start, date_end):
    """
    function to custom range a df with df input 
    """
    df = input_.STRIPS_to_df()
    start_date = pd.to_datetime(date_start)
    end_date = pd.to_datetime(date_end)
    
    # Filter dataframe between dates (inclusive)
    filtered_df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    # Sort by date (ascending - oldest first)
    filtered_df = filtered_df.sort_index()
    
    return filtered_df



def plot_arima_px(df):
    """
    Simple function to plot last Px with ARIMA prediction dots 
    """
    plt.figure(figsize=(10, 6))
    
    # Changed line color to green
    plt.plot(df.index, df['Last Px'], color='black')
    
    final_date = df.index[-1]
    final_price = df['Last Px'].iloc[-1]
    pred_dates = [final_date + pd.Timedelta(days=1), 
                  final_date + pd.Timedelta(days=7), 
                  final_date + pd.Timedelta(days=28)]
    pred_values = [54.603, 54.541, 54.517]
    
    plt.scatter(final_date, final_price, color='blue', s=40, zorder=5, 
                label=f'Last PX: {final_price:.3f}')
    
    plt.scatter(pred_dates[0], pred_values[0], color='green', s=40, zorder=5, 
                label=f'ARIMA 1 day: {pred_values[0]:.3f}')
    plt.scatter(pred_dates[1], pred_values[1], color='green', s=40, zorder=5, 
                label=f'ARIMA 7 day: {pred_values[1]:.3f}')
    plt.scatter(pred_dates[2], pred_values[2], color='green', s=40, zorder=5, 
                label=f'ARIMA 28 day: {pred_values[2]:.3f}')
    
    plt.title('Bullish ARIMA(1,1,1) Signal for SP 0 05 15 40')
    
    # Added X and Y axis titles
    plt.xlabel("Date")
    plt.ylabel("Last PX")
    
    plt.ylim(top=55)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.show()

bond_range = cap_df('2024-07-01', '2024-09-04')
print(bond_range)
#plot_arima_px(bond_range)






def plot_trades_px(df):
    """
    function to plot last Px line plot with symbols indicating buys and sells 
    """
    plt.figure(figsize=(10, 6))
    
    # Changed line color to green
    plt.plot(df.index, df['Last Px'], color='black')
    
    final_date = df.index[-1]
    final_price = df['Last Px'].iloc[-1]
    pred_dates = [final_date + pd.Timedelta(days=1), 
                  final_date + pd.Timedelta(days=7), 
                  final_date + pd.Timedelta(days=28)]
    pred_values = [54.603, 54.541, 54.517]
    
    plt.scatter(final_date, final_price, color='blue', s=40, zorder=5, 
                label=f'Last PX: {final_price:.3f}')
    
    plt.scatter(pred_dates[0], pred_values[0], color='green', s=40, zorder=5, 
                label=f'ARIMA 1 day: {pred_values[0]:.3f}')
    plt.scatter(pred_dates[1], pred_values[1], color='green', s=40, zorder=5, 
                label=f'ARIMA 7 day: {pred_values[1]:.3f}')
    plt.scatter(pred_dates[2], pred_values[2], color='green', s=40, zorder=5, 
                label=f'ARIMA 28 day: {pred_values[2]:.3f}')
    
    plt.title('Bullish ARIMA(1,1,1) Signal for SP 0 05 15 40')
    
    # Added X and Y axis titles
    plt.xlabel("Date")
    plt.ylabel("Last PX")
    
    plt.ylim(top=55)
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.legend()
    plt.show()






def plot_trading_decisions(csv_file,name,strategy, SIZE = 150):
    """
    Plot price time series with trading decisions marked
    Green diamonds = BUY, Red diamonds = SELL, Black diamonds = Close (new_position = 0)
    """
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Plot price time series
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], 'b-', linewidth=1, label='Price')
    
    # Filter trades (BUY or SELL only)
    trades = df[df['recommendation'].isin(['BUY', 'SELL'])]
    
    # Add Close markers (black diamonds) - when new_position = 0
    close_trades = trades[trades['new_position'] == 0]
    if not close_trades.empty:
        plt.scatter(close_trades['date'], close_trades['price'], 
                   color='black', marker='D', s=SIZE, zorder=5, label='Close')
    
    # Add BUY markers (green diamonds) - exclude closes
    buy_trades = trades[(trades['recommendation'] == 'BUY') & (trades['new_position'] != 0)]
    if not buy_trades.empty:
        plt.scatter(buy_trades['date'], buy_trades['price'], 
                   color='green', marker='D', s=SIZE, zorder=5, label='BUY')
    
    # Add SELL markers (red diamonds) - exclude closes
    sell_trades = trades[(trades['recommendation'] == 'SELL') & (trades['new_position'] != 0)]
    if not sell_trades.empty:
        plt.scatter(sell_trades['date'], sell_trades['price'], 
                   color='red', marker='D', s=SIZE, zorder=5, label='SELL')
    
    plt.title(f'{name} BUY/SELL/CLOSE for {strategy} Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend(markerscale=0.5)  # This scales down legend markers
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"Bond Analysis/TRADE {name} {strategy}.png")
    plt.show()



def plot_position(csv_file, name,strategy):
   """
   Plot price time series with position sizes marked
   Position size determines diamond color and size:
   1.0: Large green (100), 0.5: Small green (50), 0: Black (50), -0.5: Small red (50), -1.0: Large red (100)
   """
   # Read the CSV
   df = pd.read_csv(csv_file)
   
   # Convert date to datetime
   df['date'] = pd.to_datetime(df['date'])
   
   # Plot price time series
   plt.figure(figsize=(12, 6))
   plt.plot(df['date'], df['price'], 'b-', linewidth=1, label='Price')
   
   # Filter trades (BUY or SELL only)
   trades = df[df['recommendation'].isin(['BUY', 'SELL'])]
   
   # Position size 1.0 - Large green 
   pos_1_0 = trades[trades['new_position'] == 1.0]
   if not pos_1_0.empty:
       plt.scatter(pos_1_0['date'], pos_1_0['price'], 
                  color='green', marker="^", s=180, zorder=5, label='Long 1.0')
   
   # Position size 0.5 - Small green 
   pos_0_5 = trades[trades['new_position'] == 0.5]
   if not pos_0_5.empty:
       plt.scatter(pos_0_5['date'], pos_0_5['price'], 
                  color='green', marker='^', s=50, zorder=5, label='Long 0.5')
   
   # Position size 0 - Black 
   pos_0 = trades[trades['new_position'] == 0]
   if not pos_0.empty:
       plt.scatter(pos_0['date'], pos_0['price'], 
                  color='black', marker='o', s=50, zorder=5, label='Flat 0.0')
   
   # Position size -0.5 - Small red 
   pos_neg_0_5 = trades[trades['new_position'] == -0.5]
   if not pos_neg_0_5.empty:
       plt.scatter(pos_neg_0_5['date'], pos_neg_0_5['price'], 
                  color='red', marker='v', s=50, zorder=5, label='Short -0.5')
   
   # Position size -1.0 - Large red 
   pos_neg_1_0 = trades[trades['new_position'] == -1.0]
   if not pos_neg_1_0.empty:
       plt.scatter(pos_neg_1_0['date'], pos_neg_1_0['price'], 
                  color='red', marker='v', s=180, zorder=5, label='Short -1.0')
   
   plt.title(f'{name} Portfolio Size with {strategy} Strategy')
   plt.xlabel('Date')
   plt.ylabel('Price')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.savefig(f"Bond Analysis/Position {name} {strategy}.png")
   plt.show()




   #3 plots 

def plot_position3(csv_files, names, strategies, start_date=None, end_date=None):
   """
   Plot price time series with position sizes marked for 3 strategies
   """
   # Create figure with 3 subplots
   fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
   fig.suptitle(f'Treasury Trading Maturity Comparison: {strategies[0]} Agent', fontsize=16, y=0.98)
   
   for i, (csv_file, name, strategy) in enumerate(zip(csv_files, names, strategies)):
       

       # read the CSV
       df = pd.read_csv(csv_file)

       if i == 0 and strategies[i]== 'ARIMA Driven':  # Only for the first CSV (30-year treasury) reflect the long position for longer run data (only needed for 30 whipsaw)
           df.loc[df['date'] == '2024-09-04', ['recommendation', 'new_position']] = ['BUY', 1.0] 
           ""
       
       # Convert date to datetime
       df['date'] = pd.to_datetime(df['date'])
       
       # Filter by date range if provided
       if start_date:
           df = df[df['date'] >= pd.to_datetime(start_date)]
       if end_date:
           df = df[df['date'] <= pd.to_datetime(end_date)]
       
       # Plot price time series on the i-th subplot
       axes[i].plot(df['date'], df['price'], 'b-', linewidth=1, label='Price')
       
       # Filter trades (BUY or SELL only)
       trades = df[df['recommendation'].isin(['BUY', 'SELL'])]
       
       # Position size 1.0 - Large green 
       pos_1_0 = trades[trades['new_position'] == 1.0]
       if not pos_1_0.empty:
           axes[i].scatter(pos_1_0['date'], pos_1_0['price'], 
                         color='green', marker="^", s=180, zorder=5, label='Long 1.0')
       
       # Position size 0.5 - Small green 
       pos_0_5 = trades[trades['new_position'] == 0.5]
       if not pos_0_5.empty:
           axes[i].scatter(pos_0_5['date'], pos_0_5['price'], 
                         color='green', marker='^', s=50, zorder=5, label='Long 0.5')
       
       # Position size 0 - Black 
       pos_0 = trades[trades['new_position'] == 0]
       if not pos_0.empty:
           axes[i].scatter(pos_0['date'], pos_0['price'], 
                         color='black', marker='o', s=50, zorder=5, label='Flat 0.0')
       
       # Position size -0.5 - Small red 
       pos_neg_0_5 = trades[trades['new_position'] == -0.5]
       if not pos_neg_0_5.empty:
           axes[i].scatter(pos_neg_0_5['date'], pos_neg_0_5['price'], 
                         color='red', marker='v', s=50, zorder=5, label='Short -0.5')
       
       # Position size -1.0 - Large red 
       pos_neg_1_0 = trades[trades['new_position'] == -1.0]
       if not pos_neg_1_0.empty:
           axes[i].scatter(pos_neg_1_0['date'], pos_neg_1_0['price'], 
                         color='red', marker='v', s=180, zorder=5, label='Short -1.0')
       
       axes[i].set_title(f'{name}')
       axes[i].set_ylabel('Price')
       axes[i].legend()
       axes[i].grid(True, alpha=0.3)
   
   # Set x-axis label only on bottom plot
   axes[2].set_xlabel('Date')
   
   # Rotate x-axis labels
   plt.setp(axes[2].get_xticklabels(), rotation=45)
   
   plt.tight_layout()
   plt.savefig(f"Bond Analysis/Position Comparison {strategies[0]}.png")
   plt.show()




#  plot_position3
plot_position3(
    ['bond_trades_ARIMA_whipsaw_SP_0_05_15_30_gemini-2_5-pro.csv',
     'bond_trades_ARIMA_whipsaw_SP_0_05_15_40_gemini-2_5-pro.csv', 
     'bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro.csv'],
    ['SP 0 05 15 30', 'SP 0 05 15 40', 'SP 0 05 15 50'],
    ['ARIMA Driven', 'ARIMA Driven', 'ARIMA Driven'],
    start_date='2024-09-04',
    end_date='2024-11-04'
)

#  plot_position3
plot_position3(
    ['bond_trades_simple_SP_0_05_15_30_gemini-2_5-pro.csv',
     'bond_trades_simple_SP_0_05_15_40_gemini-2_5-pro.csv', 
     'bond_trades_simple_SP_0_05_15_50_gemini-2_5-pro.csv'],
    ['SP 0 05 15 30', 'SP 0 05 15 40', 'SP 0 05 15 50'],
    ['Autonomous', 'Autonomous', 'Autonomous'],
    start_date='2024-09-04',
    end_date='2024-11-04'
)

#trades
#plot_trading_decisions('bond_trades_ARIMA_whipsaw_SP_0_05_15_30_gemini-2_5-pro.csv',"SP 0 05 15 30","ARIMA Driven", SIZE=75)

#plot_trading_decisions('bond_trades_ARIMA_whipsaw_SP_0_05_15_40_gemini-2_5-pro.csv',"SP 0 05 15 40","ARIMA Driven")

#plot_trading_decisions('bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro.csv',"SP 0 05 15 50","ARIMA Driven")



#plot_trading_decisions('bond_trades_simple_SP_0_05_15_30_gemini-2_5-pro.csv' ,"SP 0 05 15 30","Autonomous Agent", SIZE=75)

#plot_trading_decisions('bond_trades_simple_SP_0_05_15_40_gemini-2_5-pro.csv',"SP 0 05 15 40","Autonomous Agent")

#plot_trading_decisions('bond_trades_simple_SP_0_05_15_50_gemini-2_5-pro.csv',"SP 0 05 15 50","Autonomous Agent")





#positions
#plot_position('bond_trades_ARIMA_whipsaw_SP_0_05_15_30_gemini-2_5-pro.csv',"SP 0 05 15 30","ARIMA Driven" )

#plot_position('bond_trades_ARIMA_whipsaw_SP_0_05_15_40_gemini-2_5-pro.csv', "SP 0 05 15 40","ARIMA Driven")

#plot_position('bond_trades_ARIMA_whipsaw_SP_0_05_15_50_gemini-2_5-pro.csv', "SP 0 05 15 50","ARIMA Driven")



#plot_position('bond_trades_simple_SP_0_05_15_30_gemini-2_5-pro.csv',"SP 0 05 15 30","Autonomous Agent")

#plot_position('bond_trades_simple_SP_0_05_15_40_gemini-2_5-pro.csv',"SP 0 05 15 40","Autonomous Agent")

#plot_position('bond_trades_simple_SP_0_05_15_50_gemini-2_5-pro.csv',"SP 0 05 15 50","Autonomous Agent")