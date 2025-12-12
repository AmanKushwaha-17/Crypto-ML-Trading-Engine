"""
Supabase storage for trading bot
Replaces JSON file storage with cloud database
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, date
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class SupabaseStorage:
    """Handle all database operations"""
    
    def __init__(self):
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        
        # Validate credentials
        if not supabase_url or not supabase_key:
            raise ValueError(
                "❌ Missing Supabase credentials!\n\n"
                "Add to .env file:\n"
                "SUPABASE_URL=https://your-project.supabase.co\n"
                "SUPABASE_KEY=your_service_role_key  # ⚠️ NOT anon key!\n\n"
                "Find these in: Supabase Dashboard > Settings > API"
            )
        
        # Warn if using anon key (they start with 'eyJ' and are shorter)
        if supabase_key.startswith('eyJ') and len(supabase_key) < 200:
            logger.warning(
                "⚠️ You might be using the ANON key instead of SERVICE ROLE key!\n"
                "   Service role keys are longer (300+ chars) and start with 'eyJ'\n"
                "   If you get permission errors, double-check your key."
            )
        
        try:
            self.client: Client = create_client(supabase_url, supabase_key)
            logger.info("✅ Connected to Supabase")
        except Exception as e:
            raise ConnectionError(
                f"❌ Failed to connect to Supabase: {e}\n\n"
                f"Check your SUPABASE_URL and SUPABASE_KEY in .env"
            )
        
        # Initialize bot state if it doesn't exist
        self._initialize_state()
    
    def _initialize_state(self):
        """Create bot_state record if it doesn't exist"""
        try:
            # Try to get existing state
            result = self.client.table('bot_state').select('*').eq('id', 1).execute()
            
            if not result.data:
                # Create initial state
                initial_state = {
                    'id': 1,
                    'daily_pnl': 0.0,
                    'last_reset_date': date.today().isoformat(),
                    'total_trades': 0,
                    'winning_trades': 0,
                    'is_active': True,
                    'peak_balance': 5000.0,  # Initial testnet balance
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                
                self.client.table('bot_state').insert(initial_state).execute()
                logger.info("✅ Bot state initialized in database")
            else:
                logger.info("✅ Bot state found in database")
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize state: {e}")
            logger.error("   Make sure you ran the SQL schema in Supabase!")
            raise
    
    # ==================== TRADES ====================
    
    def save_trade(self, trade: Dict) -> bool:
        """Save completed trade to database"""
        try:
            # Convert datetime objects to ISO strings
            trade_data = {
                'entry_time': trade['entry_time'].isoformat() if isinstance(trade['entry_time'], datetime) else trade['entry_time'],
                'exit_time': trade['exit_time'].isoformat() if isinstance(trade['exit_time'], datetime) else trade['exit_time'],
                'side': trade['side'],
                'direction': trade['direction'],
                'entry_price': float(trade['entry_price']),
                'exit_price': float(trade['exit_price']),
                'confidence': float(trade['confidence']),
                'leverage': int(trade['leverage']),
                'position_value': float(trade['position_value']),
                'quantity': float(trade['quantity']),
                'pnl_usd': float(trade['pnl_usd']),
                'pnl_pct': float(trade['pnl_pct']),
                'exit_reason': trade['exit_reason'],
                'duration_minutes': float(trade['duration_minutes'])
            }
            
            result = self.client.table('trades').insert(trade_data).execute()
            logger.info(f"💾 Trade saved: {trade['side']} ${trade['pnl_usd']:.2f}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to save trade: {e}")
            logger.error(f"   Trade data: {trade}")
            return False
    
    def get_all_trades(self) -> List[Dict]:
        """Get all trades from database"""
        try:
            result = self.client.table('trades').select('*').order('entry_time', desc=True).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"❌ Failed to fetch trades: {e}")
            return []
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get N most recent trades"""
        try:
            result = self.client.table('trades').select('*').order('entry_time', desc=True).limit(limit).execute()
            return result.data if result.data else []
        except Exception as e:
            logger.error(f"❌ Failed to fetch recent trades: {e}")
            return []
    
    def get_trades_count(self) -> int:
        """Get total number of trades"""
        try:
            result = self.client.table('trades').select('id', count='exact').execute()
            return result.count if result.count else 0
        except Exception as e:
            logger.error(f"❌ Failed to count trades: {e}")
            return 0
    
    # ==================== BOT STATE ====================
    
    def get_state(self) -> Dict:
        """Get current bot state"""
        try:
            result = self.client.table('bot_state').select('*').eq('id', 1).single().execute()
            
            if result.data:
                state = result.data
                return state
            else:
                logger.warning("⚠️ State not found, returning default")
                return self._default_state()
        
        except Exception as e:
            logger.error(f"❌ Failed to get state: {e}")
            return self._default_state()
    
    def update_state(self, updates: Dict) -> bool:
        """Update bot state"""
        try:
            # Ensure last_reset_date is string
            if 'last_reset_date' in updates and isinstance(updates['last_reset_date'], date):
                updates['last_reset_date'] = updates['last_reset_date'].isoformat()
            
            updates['updated_at'] = datetime.now().isoformat()
            
            result = self.client.table('bot_state').update(updates).eq('id', 1).execute()
            
            if not result.data:
                logger.error("❌ State update returned no data")
                return False
                
            logger.debug(f"✅ State updated: {list(updates.keys())}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Failed to update state: {e}")
            logger.error(f"   Updates: {updates}")
            return False
    
    def reset_daily_pnl(self) -> bool:
        """Reset daily PnL counter"""
        try:
            updates = {
                'daily_pnl': 0.0,
                'last_reset_date': date.today().isoformat()
            }
            return self.update_state(updates)
        except Exception as e:
            logger.error(f"❌ Failed to reset daily PnL: {e}")
            return False
    
    def increment_trade_counters(self, is_win: bool) -> bool:
        """Increment trade counters after closing position"""
        try:
            state = self.get_state()
            updates = {
                'total_trades': state['total_trades'] + 1,
                'winning_trades': state['winning_trades'] + (1 if is_win else 0)
            }
            return self.update_state(updates)
        except Exception as e:
            logger.error(f"❌ Failed to increment counters: {e}")
            return False
    
    def add_to_daily_pnl(self, pnl: float) -> bool:
        """Add trade PnL to daily total"""
        try:
            state = self.get_state()
            updates = {
                'daily_pnl': state['daily_pnl'] + pnl
            }
            return self.update_state(updates)
        except Exception as e:
            logger.error(f"❌ Failed to update daily PnL: {e}")
            return False
    
    def update_peak_balance(self, balance: float) -> bool:
        """Update peak balance if new high"""
        try:
            state = self.get_state()
            if balance > state['peak_balance']:
                return self.update_state({'peak_balance': balance})
            return True
        except Exception as e:
            logger.error(f"❌ Failed to update peak balance: {e}")
            return False
    
    def _default_state(self) -> Dict:
        """Return default state"""
        return {
            'id': 1,
            'daily_pnl': 0.0,
            'last_reset_date': date.today().isoformat(),
            'total_trades': 0,
            'winning_trades': 0,
            'is_active': True,
            'peak_balance': 5000.0
        }
    
    # ==================== ANALYTICS ====================
    
    def get_performance_summary(self) -> Dict:
        """Get quick performance stats"""
        try:
            trades = self.get_all_trades()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'total_pnl': 0,
                    'avg_pnl': 0
                }
            
            total_trades = len(trades)
            winning_trades = sum(1 for t in trades if t['pnl_usd'] > 0)
            total_pnl = sum(t['pnl_usd'] for t in trades)
            
            return {
                'total_trades': total_trades,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'total_pnl': total_pnl,
                'avg_pnl': total_pnl / total_trades if total_trades > 0 else 0
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to get summary: {e}")
            return {'error': str(e)}
    
    # ==================== HEALTH CHECK ====================
    
    def health_check(self) -> bool:
        """Verify database connection and tables"""
        try:
            # Test connection and tables
            state = self.get_state()
            count = self.get_trades_count()
            
            logger.info("✅ Database health check passed")
            logger.info(f"   - Bot state: Active={state.get('is_active', False)}")
            logger.info(f"   - Total trades: {count}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Database health check failed: {e}")
            logger.error("   Make sure:")
            logger.error("   1. You ran the SQL schema in Supabase")
            logger.error("   2. You're using the SERVICE ROLE key (not anon key)")
            logger.error("   3. Tables 'trades' and 'bot_state' exist")
            return False