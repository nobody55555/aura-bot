class RiskManager:
    RISK_PER_TRADE = 0.02  # 2%
    RR_RATIO = 3  # 3:1

    def position_size(self, balance: float, entry_price: float, stop_loss: float) -> float:
        risk_amount = balance * self.RISK_PER_TRADE
        risk_per_unit = abs(entry_price - stop_loss)
        return risk_amount / risk_per_unit if risk_per_unit > 0 else 0

    def set_orders(self, ex, symbol, size, entry_price):
        stop_loss = entry_price * 0.98
        take_profit = entry_price * (1 + self.RR_RATIO * 0.02)
        # ex.create_stop_loss_order(symbol, size, stop_loss)  # Für Live
        # ex.create_take_profit_order(symbol, size, take_profit)
        return stop_loss, take_profit
