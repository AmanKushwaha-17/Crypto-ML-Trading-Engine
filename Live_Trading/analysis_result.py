"""
Analyze hybrid bot performance
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def load_trades(filepath='hybrid_trades.json'):
    """Load trade history"""
    try:
        with open(filepath, 'r') as f:
            trades = json.load(f)
        return pd.DataFrame(trades)
    except FileNotFoundError:
        print(f"No trades file found at {filepath}")
        return pd.DataFrame()


def calculate_metrics(df):
    """Calculate performance metrics"""
    if df.empty:
        print("No trades to analyze")
        return
    
    # Convert timestamps
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])
    
    # Basic stats
    total_trades = len(df)
    winning_trades = (df['pnl_usd'] > 0).sum()
    losing_trades = (df['pnl_usd'] < 0).sum()
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    # PnL stats
    total_pnl = df['pnl_usd'].sum()
    avg_win = df[df['pnl_usd'] > 0]['pnl_usd'].mean() if winning_trades > 0 else 0
    avg_loss = df[df['pnl_usd'] < 0]['pnl_usd'].mean() if losing_trades > 0 else 0
    
    # Risk metrics
    max_win = df['pnl_usd'].max()
    max_loss = df['pnl_usd'].min()
    
    # Profit factor
    gross_profit = df[df['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(df[df['pnl_usd'] < 0]['pnl_usd'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    # Sharpe ratio (annualized)
    returns = df['pnl_pct'] / 100
    sharpe = returns.mean() / returns.std() * np.sqrt(365*24/1) if len(returns) > 1 else 0
    
    # Max drawdown
    cumulative = df['pnl_usd'].cumsum()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max)
    max_drawdown = drawdown.min()
    max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0
    
    # Long vs Short performance
    long_trades = df[df['side'] == 'LONG']
    short_trades = df[df['side'] == 'SHORT']
    
    long_win_rate = (long_trades['pnl_usd'] > 0).sum() / len(long_trades) * 100 if len(long_trades) > 0 else 0
    short_win_rate = (short_trades['pnl_usd'] > 0).sum() / len(short_trades) * 100 if len(short_trades) > 0 else 0
    
    # Print results
    print("="*60)
    print("HYBRID BOT PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"\n📊 TRADE STATISTICS")
    print(f"Total Trades: {total_trades}")
    print(f"Winning Trades: {winning_trades} ({win_rate:.1f}%)")
    print(f"Losing Trades: {losing_trades}")
    print(f"\n💰 PNL METRICS")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Avg Win: ${avg_win:.2f}")
    print(f"Avg Loss: ${avg_loss:.2f}")
    print(f"Best Trade: ${max_win:.2f}")
    print(f"Worst Trade: ${max_loss:.2f}")
    print(f"\n📈 RISK METRICS")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: ${max_drawdown:.2f} ({max_drawdown_pct:.1f}%)")
    print(f"\n🎯 DIRECTION BREAKDOWN")
    print(f"Long Trades: {len(long_trades)} (Win Rate: {long_win_rate:.1f}%)")
    print(f"Short Trades: {len(short_trades)} (Win Rate: {short_win_rate:.1f}%)")
    print(f"\n⏱️ TIMING")
    print(f"First Trade: {df['entry_time'].min()}")
    print(f"Last Trade: {df['exit_time'].max()}")
    print(f"Avg Duration: {df['duration_minutes'].mean():.0f} minutes")
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'profit_factor': profit_factor,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }


def plot_performance(df):
    """Create performance visualizations"""
    if df.empty or len(df) < 2:
        print("Not enough data to plot")
        return
    
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df = df.sort_values('entry_time')
    
    # Calculate cumulative PnL
    df['cumulative_pnl'] = df['pnl_usd'].cumsum()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hybrid Bot Performance', fontsize=16, fontweight='bold')
    
    # 1. Cumulative PnL
    ax1 = axes[0, 0]
    ax1.plot(df['entry_time'], df['cumulative_pnl'], linewidth=2, color='blue')
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax1.set_title('Cumulative PnL Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('PnL (USD)')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Trade Distribution
    ax2 = axes[0, 1]
    wins = df[df['pnl_usd'] > 0]['pnl_usd']
    losses = df[df['pnl_usd'] < 0]['pnl_usd']
    ax2.hist([wins, losses], bins=20, label=['Wins', 'Losses'], color=['green', 'red'], alpha=0.7)
    ax2.set_title('PnL Distribution')
    ax2.set_xlabel('PnL (USD)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Win Rate by Confidence
    ax3 = axes[1, 0]
    df['confidence_bin'] = pd.cut(df['confidence'], bins=5)
    win_by_conf = df.groupby('confidence_bin').apply(
        lambda x: (x['pnl_usd'] > 0).sum() / len(x) * 100
    )
    win_by_conf.plot(kind='bar', ax=ax3, color='purple', alpha=0.7)
    ax3.set_title('Win Rate by Confidence Level')
    ax3.set_xlabel('Confidence Range')
    ax3.set_ylabel('Win Rate (%)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Long vs Short Performance
    ax4 = axes[1, 1]
    side_pnl = df.groupby('side')['pnl_usd'].sum()
    colors = ['green' if x > 0 else 'red' for x in side_pnl]
    side_pnl.plot(kind='bar', ax=ax4, color=colors, alpha=0.7)
    ax4.set_title('Total PnL by Direction')
    ax4.set_xlabel('Side')
    ax4.set_ylabel('Total PnL (USD)')
    ax4.tick_params(axis='x', rotation=0)
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hybrid_bot_performance.png', dpi=300, bbox_inches='tight')
    print("\n📊 Performance chart saved: hybrid_bot_performance.png")
    plt.show()


def export_summary(df, metrics, filepath='performance_summary.txt'):
    """Export text summary"""
    with open(filepath, 'w') as f:
        f.write("HYBRID BOT PERFORMANCE SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Trades: {metrics['total_trades']}\n")
        f.write(f"Win Rate: {metrics['win_rate']:.1f}%\n")
        f.write(f"Total PnL: ${metrics['total_pnl']:,.2f}\n")
        f.write(f"Profit Factor: {metrics['profit_factor']:.2f}\n")
        f.write(f"Sharpe Ratio: {metrics['sharpe']:.2f}\n")
        f.write(f"Max Drawdown: ${metrics['max_drawdown']:.2f}\n\n")
        
        f.write("RECENT TRADES (Last 10)\n")
        f.write("-"*60 + "\n")
        for _, trade in df.tail(10).iterrows():
            f.write(f"{trade['entry_time']} | {trade['side']:5} | "
                   f"${trade['pnl_usd']:+7.2f} | {trade['pnl_pct']:+6.2f}%\n")
    
    print(f"📄 Summary exported: {filepath}")


if __name__ == "__main__":
    # Load and analyze
    df = load_trades()
    
    if not df.empty:
        metrics = calculate_metrics(df)
        
        # Ask user if they want plots
        try:
            response = input("\n📊 Generate performance charts? (y/n): ").lower()
            if response == 'y':
                plot_performance(df)
        except:
            pass
        
        # Export summary
        export_summary(df, metrics)
        
        print("\n✅ Analysis complete!")
    else:
        print("❌ No trade data found. Run the bot first!")