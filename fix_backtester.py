# fix_backtester.py
import re

file_path = r'E:\Bot_FVG\trading_bot\core\backtest\backtester.py'

# Read file
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace all self.config.xxx with self.config['xxx']
replacements = {
    'self.config.initial_balance': "self.config['initial_balance']",
    'self.config.risk_per_trade': "self.config['risk_per_trade']",
    'self.config.base_lot_size': "self.config['base_lot_size']",
    'self.config.pip_value': "self.config['pip_value']",
    'self.config.commission_per_lot': "self.config['commission_per_lot']",
    'self.config.consecutive_losses_trigger': "self.config['consecutive_losses_trigger']",
    'self.config.martingale_multiplier': "self.config['martingale_multiplier']",
    'self.config.max_lot_size': "self.config['max_lot_size']",
    'self.config.min_confidence_score': "self.config['min_confidence_score']",
    'self.config.atr_sl_multiplier': "self.config['atr_sl_multiplier']",
    'self.config.atr_tp_multiplier': "self.config['atr_tp_multiplier']",
}

for old, new in replacements.items():
    content = content.replace(old, new)

# Write back
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed backtester.py!")