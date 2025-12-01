NUMBERS=set(range(37)) #roulette number
NUM_SLOT=37
OMEGA_MEAN = 15.0
OMEGA_STD = 2.0
TSPIN_LAMBDA = 0.8
WHEEL = [0,32,15,19,4,21,2,25,
        17,34,6,27,13,36,11,30,
        8,23,10,5,24,16,33,1,20,14,
        31,9,22,18,29,7,28,12,35,3,26] #roulette order

#Outside Bets
COLOR_MAP= {
    "1": {1,3,5,7,9,12,14,16,18,19,21,23,25,27,30,32,34,36}, #red
    "2": {2,4,6,8,10,11,13,15,17,20,22,24,26,28,29,31,33,35}, #black
    "3": {0} #green
}
DOZEN_MAP ={
    "1": set(range(1, 13)),    # 1..12
    "2": set(range(13, 25)),   # 13..24
    "3": set(range(25, 37))    # 25..36
}

ODD_EVEN = {
    "1": {n for n in NUMBERS if n % 2 == 1}, #odd
    "2": {n for n in NUMBERS if n % 2 == 0} #even
}
LOW_HIGH={
    "1": set(range(1, 19)), #low
    "2": set(range(19, 37)) #high

}
COLUMNS = {
    "1": {1,4,7,10,13,16,19,22,25,28,31,34},
    "2": {2,5,8,11,14,17,20,23,26,29,32,35},
    "3": {3,6,9,12,15,18,21,24,27,30,33,36}
}

BETS={
    "1": "color",
    "2": "odd_even",
    "3": "low_high",
    "4": "columns",
    "5": "dozen",
    "6": "number"
}