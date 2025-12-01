from .bet import Bet
from . import config
from .spin_engine import SpinEngine 
from .rng import RNG

class BettingEngine:

    def __init__(self,config):
        self.config = config
        self.rng = RNG()
        self.spin = SpinEngine(self.rng,config)

    def run_bet(self):
        try:
            bet_value= int(input("Place your bet: "))
        except ValueError:
            print("It has to be a number!")
            return
        else:
            print("Choose bet type:")
            print("1) Color (red/black/green)")
            print("2) Odd/Even")
            print("3) Low/High")
            print("4) Columns")
            print("5) Dozen (1-12, 13-24, 25-36)")
            print("6) Single number (0-36)")
            choice = input("> ").strip()
            if choice == "1":
                bet_type = input("Enter color:\n[1]red\n[2]black\n"
                "[3]green\n>").strip().lower()
                if bet_type not in self.config.COLOR_MAP:
                    print("Invalid color")
                    return
                bet=Bet(self.config.BETS[choice],bet_value, self.config.COLOR_MAP[bet_type])
            elif choice == "2":
                bet_type = input("Choose:\n [1]odd\n [2]even\n>")
                if bet_type not in self.config.ODD_EVEN:
                    print("Invalid argumment")
                    return
                bet=Bet(self.config.BETS[choice],bet_value, self.config.ODD_EVEN[bet_type])
            elif choice == "3":
                bet_type = input("Choose:\n[1]low\n[2]high\n>")
                if bet_type not in self.config.LOW_HIGH:
                    print("Invalid dozen")
                    return
                bet=Bet(self.config.BETS[choice], bet_value, self.config.LOW_HIGH[bet_type])
            elif choice == "4":
                bet_type = input("Choose columns[1][2][3]\n>")
                if bet_type not in self.config.LOW_HIGH:
                    print("Invalid dozen")
                    return
                bet=Bet(self.config.BETS[choice], bet_value, self.config.COLUMNS[bet_type])
            elif choice == "5":
                bet_type = input("Choose dozen:\n[1] = 1-12\n"
                "[2] = 13-24\n[3] = 25-36\n>")
                if bet_type not in self.config.DOZEN_MAP:
                    print("Invalid dozen")
                    return
                bet=Bet(self.config.BETS[choice], bet_value, self.config.DOZEN_MAP[bet_type])
            elif choice == "6":
                try:
                    bet_type = int(input("Enter a number: "))
                except ValueError:
                    print("That’s not a number!")
                    return
                else:
                    if bet_type not in self.config.WHEEL:
                        print("That number is not on the wheel.")
                        return
                    bet=Bet(self.config.BETS[choice], bet_value, bet_type)
            print(f"Bet accepted: {bet}")
            slot = self.spin.simulate()
            self.evaluate_bet(slot, bet)

    def evaluate_bet(self, slot, bet):
        print(slot)
        target = bet.targets
        if bet.bet_type != "number":
            if slot in target:
                print("YOU WON!!🎉")
                return
            else:
                print("You lost better luck next time😕")
                return
        else:
            if slot == target:
                print("YOU WON!!🎉")
                return
            else:
                print("Better luck next time😕")
                return



if __name__ == "__main__":
    #from .rng import RNG
    from . import config

    engine = BettingEngine(config)
    engine.run_bet()