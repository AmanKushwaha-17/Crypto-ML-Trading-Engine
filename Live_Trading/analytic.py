# Add this after imports, before Config class
import math
import pandas as pd
from typing import Dict
from collections import defaultdict

class PerformanceAnalytics:
    """Calculate trading performance metrics"""
    
    def __init__(self):
        self.trades = []
    
    def add_trade(self, trade: Dict):
        """Add completed trade to history"""
        self.trades.append(trade)
    
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return self._empty_metrics()
        
        df_trades = pd.DataFrame(self.trades)
        
        # Basic stats
        total_trades = len(df_trades)
        winning_trades = (df_trades['pnl_usd'] > 0).sum()
        losing_trades = (df_trades['pnl_usd'] < 0).sum()
        breakeven_trades = (df_trades['pnl_usd'] == 0).sum()
        
        # Win rate
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = df_trades['pnl_usd'].sum()
        avg_win = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].mean() if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades['pnl_usd'] < 0]['pnl_usd'].mean() if losing_trades > 0 else 0
        
        # Best and worst
        max_win = df_trades['pnl_usd'].max()
        max_loss = df_trades['pnl_usd'].min()
        
        # Profit factor
        gross_profit = df_trades[df_trades['pnl_usd'] > 0]['pnl_usd'].sum()
        gross_loss = abs(df_trades[df_trades['pnl_usd'] < 0]['pnl_usd'].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Sharpe Ratio (annualized)
        returns = df_trades['pnl_pct'] / 100
        sharpe_ratio = self._calculate_sharpe(returns)
        
        # Max drawdown
        cumulative_pnl = df_trades['pnl_usd'].cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = cumulative_pnl - running_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = self._max_consecutive(df_trades['pnl_usd'] > 0)
        consecutive_losses = self._max_consecutive(df_trades['pnl_usd'] < 0)
        
        # Long vs Short performance
        long_trades = df_trades[df_trades['side'] == 'LONG']
        short_trades = df_trades[df_trades['side'] == 'SHORT']
        
        long_win_rate = (long_trades['pnl_usd'] > 0).sum() / len(long_trades) * 100 if len(long_trades) > 0 else 0
        short_win_rate = (short_trades['pnl_usd'] > 0).sum() / len(short_trades) * 100 if len(short_trades) > 0 else 0
        
        long_pnl = long_trades['pnl_usd'].sum() if len(long_trades) > 0 else 0
        short_pnl = short_trades['pnl_usd'].sum() if len(short_trades) > 0 else 0
        
        # Average hold time
        avg_duration = df_trades['duration_minutes'].mean()
        
        # Expectancy (average $ per trade)
        expectancy = total_pnl / total_trades if total_trades > 0 else 0
        
        return {
            # Basic
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'breakeven_trades': breakeven_trades,
            'win_rate': win_rate,
            
            # PnL
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'max_win': max_win,
            'max_loss': max_loss,
            'expectancy': expectancy,
            
            # Risk metrics
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            
            # Streaks
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            
            # Direction breakdown
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_win_rate': long_win_rate,
            'short_win_rate': short_win_rate,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            
            # Other
            'avg_duration_minutes': avg_duration,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss
        }
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """
        Calculate annualized Sharpe Ratio
        Assumes 15-min trading intervals, 24/7 market
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return == 0:
            return 0.0
        
        # Annualization factor for 15-min bars: sqrt(365 * 24 * 4)
        # 365 days * 24 hours * 4 (15-min periods per hour) = 35,040 periods/year
        annualization_factor = math.sqrt(35040)
        
        sharpe = (mean_return / std_return) * annualization_factor
        return sharpe
    
    def _max_consecutive(self, condition: pd.Series) -> int:
        """Calculate maximum consecutive True values"""
        max_streak = 0
        current_streak = 0
        
        for value in condition:
            if value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'breakeven_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_win': 0,
            'max_loss': 0,
            'expectancy': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'long_trades': 0,
            'short_trades': 0,
            'long_win_rate': 0,
            'short_win_rate': 0,
            'long_pnl': 0,
            'short_pnl': 0,
            'avg_duration_minutes': 0,
            'gross_profit': 0,
            'gross_loss': 0
        }
    
    def print_summary(self):
        """Print formatted performance summary"""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*70)
        print("📊 PERFORMANCE SUMMARY")
        print("="*70)
        
        print(f"\n📈 OVERALL PERFORMANCE")
        print(f"  Total Trades:        {metrics['total_trades']}")
        print(f"  Winning Trades:      {metrics['winning_trades']} ({metrics['win_rate']:.1f}%)")
        print(f"  Losing Trades:       {metrics['losing_trades']}")
        print(f"  Total PnL:           ${metrics['total_pnl']:,.2f}")
        print(f"  Expectancy:          ${metrics['expectancy']:.2f} per trade")
        
        print(f"\n💰 PROFIT/LOSS METRICS")
        print(f"  Gross Profit:        ${metrics['gross_profit']:,.2f}")
        print(f"  Gross Loss:          ${metrics['gross_loss']:,.2f}")
        print(f"  Average Win:         ${metrics['avg_win']:.2f}")
        print(f"  Average Loss:        ${metrics['avg_loss']:.2f}")
        print(f"  Best Trade:          ${metrics['max_win']:.2f}")
        print(f"  Worst Trade:         ${metrics['max_loss']:.2f}")
        
        print(f"\n📊 RISK METRICS")
        print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:        ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.1f}%)")
        
        print(f"\n🔥 STREAKS")
        print(f"  Max Consecutive Wins:   {metrics['max_consecutive_wins']}")
        print(f"  Max Consecutive Losses: {metrics['max_consecutive_losses']}")
        
        print(f"\n🎯 DIRECTION BREAKDOWN")
        print(f"  LONG Trades:         {metrics['long_trades']} (Win Rate: {metrics['long_win_rate']:.1f}%)")
        print(f"  LONG PnL:            ${metrics['long_pnl']:,.2f}")
        print(f"  SHORT Trades:        {metrics['short_trades']} (Win Rate: {metrics['short_win_rate']:.1f}%)")
        print(f"  SHORT PnL:           ${metrics['short_pnl']:,.2f}")
        
        print(f"\n⏱️  TIMING")
        print(f"  Avg Hold Time:       {metrics['avg_duration_minutes']:.1f} minutes")
        
        print("="*70 + "\n")