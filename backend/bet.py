class Bet:
    def __init__(self, bet_type, amount, targets):
        #self.player_id = player_id
        self.bet_type = bet_type
        self.amount = amount
        self.targets = targets 
    
    def __str__(self):
        return f"Bet(type={self.bet_type}, amount={self.amount}, targets={self.targets})"

    def __repr__(self):
        return self.__str__()