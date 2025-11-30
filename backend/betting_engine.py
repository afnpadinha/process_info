#from .rng import RNG
from . import config
DOZEN_MAP ={
    "1": range(1,12),
    "2": range(14,24),
    "3": range(25,36)
}
NUMBERS=set(range(37))
ODD_EVEN = {
    "1": {n for n in NUMBERS if n % 2 == 1},
    "2": {n for n in NUMBERS if n % 2 == 0}
}
class BettingEngine:

    def __init__(self,config):
        self.config = config

    def run_bet(self):
        print("Choose bet type:")
        print("1) Color (red/black/green)")
        print("2) Odd/Even")
        print("3) Dozen (1-12, 13-24, 25-36)")
        print("4) Single number (0-36)")
        choice = input("> ").strip()
        if choice == "1":
            bet = input("Enter color:\n[1]red\n[2]black\n"
            "[3]green\n>").strip().lower()
            if bet not in self.config.COLOR_MAP:
                print("Invalid color")
                return
            labels={"1":"red", "2":"black", "3":"green"}
            print(f"Bet accepted: {labels.get(bet, 'unkown').upper()}")
        elif choice == "2":
            bet = input("CHoose:\n [1]odd\n [2]even\n>")
            if bet not in ODD_EVEN:
                print("Invalid argumment")
                return
            labels={"1":"odd", "2":"even"}
            print(f"Bet accepted: {labels.get(bet, 'unkown').upper()}")
        elif choice == "3":
            bet = input("Choose dozen:\n[1] = 1-12\n"
            "[2] = 13-24\n[3] = 25-36\n>")
            if bet not in DOZEN_MAP:
                print("Invalid dozen")
                return
            print(f"Bet accepted: dozen {bet}")
        elif choice == "4":
            try:
                bet = int(input("Enter a number: "))
            except ValueError:
                print("That’s not a number!")
                return
            else:
                if bet not in self.config.WHEEL:
                    print("That number is not on the wheel.")
                    return
                print(f"Bet accepted: {bet}")
                 
if __name__ == "__main__":
    #from .rng import RNG
    from . import config

    engine = BettingEngine(config)
    engine.run_bet()